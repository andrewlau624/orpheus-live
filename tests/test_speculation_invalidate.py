"""Tests for the Speculator: stale generations discarded, matches reused, sentence hook."""

import threading
from collections.abc import Iterator

from orpheus_live.core.speculation import Speculator


class BlockingGenerate:
    """A fake generate_stream() whose completion the test controls per-basis.

    Yields the reply word-by-word (so first-sentence detection can fire mid-stream);
    blocks on a per-basis gate before producing anything.
    """

    def __init__(self):
        self.calls: list[str] = []
        self.gates: dict[str, threading.Event] = {}
        self.replies: dict[str, str] = {}

    def add(self, basis: str, reply: str) -> threading.Event:
        gate = threading.Event()
        self.gates[basis] = gate
        self.replies[basis] = reply
        return gate

    def __call__(self, text: str) -> Iterator[str]:
        self.calls.append(text)
        gate = self.gates.get(text)
        if gate is not None:
            gate.wait(timeout=5)
        for word in self.replies.get(text, f"reply to: {text}").split(" "):
            yield word + " "


def _first_sentence(text: str) -> str | None:
    i = text.find(".")
    return text[: i + 1].strip() if i != -1 else None


def test_matching_final_reuses_speculative_reply():
    gen = BlockingGenerate()
    gen.add("hello there", "hi yourself").set()  # completes immediately
    spec = Speculator(gen)

    spec.on_partial("hello there")
    assert spec.take("Hello there.", timeout=2) == "hi yourself"


def test_unchanged_partial_does_not_restart_generation():
    gen = BlockingGenerate()
    gen.add("hello there", "hi").set()
    spec = Speculator(gen)

    spec.on_partial("hello there")
    spec.on_partial("Hello there.")  # same words, different punctuation/case
    spec.on_partial("hello THERE")

    assert len(gen.calls) == 1


def test_diverging_partial_discards_stale_generation():
    gen = BlockingGenerate()
    slow_gate = gen.add("hello", "stale reply")  # first gen: blocked
    gen.add("hello can you help me", "fresh reply").set()
    spec = Speculator(gen)

    spec.on_partial("hello")
    spec.on_partial("hello can you help me")  # diverges -> restart
    slow_gate.set()  # let the stale generation finish late

    assert spec.take("hello can you help me", timeout=2) == "fresh reply"


def test_diverged_final_returns_none():
    gen = BlockingGenerate()
    gen.add("hello", "reply").set()
    spec = Speculator(gen)

    spec.on_partial("hello")
    assert spec.take("hello and also something else", timeout=1) is None


def test_reset_invalidates_in_flight_generation():
    gen = BlockingGenerate()
    gen.add("hello", "reply").set()
    spec = Speculator(gen)

    spec.on_partial("hello")
    spec.reset()
    assert spec.take("hello", timeout=1) is None


def test_take_without_any_speculation_returns_none():
    spec = Speculator(lambda t: iter(()))
    assert spec.take("anything", timeout=0.1) is None


def test_first_sentence_hook_fires_once_when_a_sentence_completes():
    fired: list[str] = []
    gen = BlockingGenerate()
    gen.add("tell me about jazz", "I love jazz. It is so soulful and warm.").set()
    spec = Speculator(gen, on_first_sentence=fired.append, first_sentence=_first_sentence)

    spec.on_partial("tell me about jazz")
    assert spec.take("Tell me about jazz.", timeout=2) == "I love jazz. It is so soulful and warm."
    assert fired == ["I love jazz."]  # fired exactly once, with the first sentence


def test_first_sentence_hook_not_fired_for_superseded_generation():
    fired: list[str] = []
    gen = BlockingGenerate()
    slow_gate = gen.add("hello", "Stale sentence here. More.")  # blocked
    gen.add("hello there friend", "Fresh sentence here. More.").set()
    spec = Speculator(gen, on_first_sentence=fired.append, first_sentence=_first_sentence)

    spec.on_partial("hello")
    spec.on_partial("hello there friend")  # supersedes the first gen
    result = spec.take("hello there friend", timeout=2)
    slow_gate.set()  # stale gen finishes late -> must NOT fire the hook

    assert result == "Fresh sentence here. More."
    assert fired == ["Fresh sentence here."]
