"""Thinking-while-listening: speculative reply generation AND turn-end prediction.

While the user is still talking, the conversation loop periodically feeds the
in-progress transcript here. Two speculative tracks run off those partials:

- `Speculator` generates the *reply text* in the background (tagged with a
  generation id); stale generations discard their own results. When the user
  stops, if the final transcript still matches the in-flight basis, the reply is
  reused as-is -- otherwise the caller regenerates synchronously.
- `TurnPredictor` asks the cognition model, for each partial, "if they pause
  right after this, is their thought complete?" -- so when a real pause lands,
  the *model's* turn-taking verdict is already cached and costs ~0ms to read.
  This is what lets turn-taking be model-driven without putting a seconds-slow
  model on the reply-latency-critical path: predict during speech, not after it.

Streaming: `generate_stream` yields reply chunks as they arrive. As soon as the
first *complete sentence* is detected, `on_first_sentence` fires (once per
generation) so pre-synthesis can start immediately -- well before the user even
stops talking. `take()` returns the full aggregated reply on a hit.
"""

import re
import threading
import time
from collections.abc import Callable, Iterator

from ..debug import tracer
from ..models import CognitionDecision

_NORMALIZE = re.compile(r"[^a-z0-9 ]+")


def _norm(text: str) -> str:
    return _NORMALIZE.sub("", text.lower()).strip()


class Speculator:
    """Runs at-most-one live speculative generation, keyed by transcript prefix.

    `generate_stream` is injected (Brain.generate_stream in production, a fake in
    tests) and must NOT mutate conversation history -- committing the turn is the
    caller's job, only once a reply is actually used. `first_sentence` extracts the
    first complete sentence from partial text (or None); `on_first_sentence` fires
    once per generation the moment that sentence is ready.
    """

    def __init__(
        self,
        generate_stream: Callable[[str], Iterator[str]],
        on_first_sentence: Callable[[str], None] | None = None,
        first_sentence: Callable[[str], str | None] | None = None,
    ):
        self._generate_stream = generate_stream
        self._on_first_sentence = on_first_sentence
        self._first_sentence = first_sentence
        self._lock = threading.Lock()
        self._gen_id = 0
        self._basis = ""  # normalized transcript the current generation is based on
        self._reply = ""  # aggregated reply text so far
        self._done = threading.Event()

    def reset(self) -> None:
        """Invalidate any in-flight generation (call when the AI takes the turn)."""
        with self._lock:
            self._gen_id += 1
            self._basis = ""
            self._reply = ""
            self._done.set()  # release any waiter; stale gen_id means it gets None

    def on_partial(self, text: str) -> None:
        """Feed the latest partial transcript; (re)starts generation if it changed."""
        norm = _norm(text)
        if not norm:
            return
        with self._lock:
            if norm == self._basis:
                return  # transcript hasn't meaningfully changed -> keep current gen
            self._gen_id += 1
            my_gen = self._gen_id
            self._basis = norm
            self._reply = ""
            self._done.clear()
        tracer.emit("spec.bet", _echo=False, text=text)
        threading.Thread(target=self._run, args=(my_gen, text), daemon=True).start()

    def _run(self, my_gen: int, text: str) -> None:
        acc: list[str] = []
        fired = False
        try:
            for chunk in self._generate_stream(text):
                with self._lock:
                    if my_gen != self._gen_id:
                        return  # superseded -> drop
                    self._reply += chunk
                acc.append(chunk)
                if not fired and self._first_sentence and self._on_first_sentence:
                    sentence = self._first_sentence("".join(acc))
                    if sentence:
                        fired = True
                        with self._lock:
                            if my_gen != self._gen_id:
                                return
                        try:
                            self._on_first_sentence(sentence)
                        except Exception:
                            pass
        except Exception:
            pass
        with self._lock:
            if my_gen == self._gen_id:
                self._done.set()

    def take(self, final_text: str, timeout: float) -> str | None:
        """Return the speculative reply if it was based on `final_text`, else None.

        Waits up to `timeout` for an in-flight generation to finish. A mismatch
        between the final transcript and the current basis means the user's
        words diverged from what we bet on, so the caller must regenerate.
        """
        with self._lock:
            if _norm(final_text) != self._basis or not self._basis:
                return None
        if not self._done.wait(timeout):
            return None
        with self._lock:
            return self._reply.strip() or None


class TurnPredictor:
    """Pre-computes the turn-taking verdict while the user is still talking.

    For each partial transcript, a background thread asks the cognition model
    "if they pause right after this, is their thought complete — do I take the
    turn?". The verdict is cached keyed to the transcript it judged. When a real
    pause fires, `verdict_for()` returns the model's decision instantly (or None
    if the text diverged / the judgment is still in flight — the caller falls
    back to waiting for the turn-end safety net, so a slow model can only ever
    delay to the net's timeout, never beyond it).

    Same generation-id discipline as Speculator: at most one live judgment;
    stale threads discard their own results.
    """

    def __init__(self, decide: Callable[[str], CognitionDecision]):
        # `decide` is injected: production passes a decide_turn(...) closure, tests a fake.
        self._decide = decide
        self._lock = threading.Lock()
        self._gen_id = 0
        self._basis = ""  # normalized transcript the current judgment is based on
        self._verdict: CognitionDecision | None = None

    def reset(self) -> None:
        """Invalidate any in-flight judgment (call when the AI takes the turn)."""
        with self._lock:
            self._gen_id += 1
            self._basis = ""
            self._verdict = None

    def on_partial(self, text: str) -> None:
        """Feed the latest partial transcript; (re)judges the turn if it changed."""
        norm = _norm(text)
        if not norm:
            return
        with self._lock:
            if norm == self._basis:
                return  # transcript hasn't meaningfully changed -> keep current verdict
            self._gen_id += 1
            my_gen = self._gen_id
            self._basis = norm
            self._verdict = None
        threading.Thread(target=self._run, args=(my_gen, text), daemon=True).start()

    def _run(self, my_gen: int, text: str) -> None:
        t0 = time.monotonic()
        try:
            decision = self._decide(text)
        except Exception:
            return  # model hiccup -> no verdict; the safety net covers the pause
        with self._lock:
            if my_gen != self._gen_id:
                return  # superseded while judging -> drop
            self._verdict = decision
        tracer.emit(
            "turn.predict",
            action=decision.action.name,
            wall_s=time.monotonic() - t0,
            text=text,
        )

    def verdict_for(self, final_text: str) -> CognitionDecision | None:
        """The cached model verdict for `final_text`, or None. Never blocks."""
        with self._lock:
            if not self._basis or _norm(final_text) != self._basis:
                return None
            return self._verdict
