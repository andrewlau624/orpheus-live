"""Thinking-while-listening: speculatively generate a reply from partial transcripts.

While the user is still talking, the conversation loop periodically feeds the
in-progress transcript here. Each time the transcript meaningfully changes we
start generating a fresh reply in the background (tagged with a generation id);
stale generations discard their own results. When the user finally stops, if
the final transcript still matches what the in-flight generation was based on,
its reply is reused as-is -- the "reply nearly ready when you stop talking"
behavior -- otherwise we fall back to a fresh synchronous generation.

Streaming: `generate_stream` yields reply chunks as they arrive. As soon as the
first *complete sentence* is detected, `on_first_sentence` fires (once per
generation) so pre-synthesis can start immediately -- well before the user even
stops talking. `take()` returns the full aggregated reply on a hit.
"""

import re
import threading
from collections.abc import Callable, Iterator

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
