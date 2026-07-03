"""Tests for TurnPredictor: model-driven turn-end verdicts pre-computed during speech.

Mirrors the Speculator's generation-id discipline tests: verdicts are keyed to the
transcript they judged; stale judgments discard themselves; reset invalidates.
"""

import threading

from orpheus_live.core.speculation import TurnPredictor
from orpheus_live.models import CognitionAction, CognitionDecision


def _decision(action=CognitionAction.SPEAK, thought="they're done"):
    return CognitionDecision(action=action, urgency=0.7, thought=thought)


def _wait_for(predicate, timeout=2.0):
    done = threading.Event()

    def poll():
        import time

        while not predicate():
            time.sleep(0.005)
        done.set()

    threading.Thread(target=poll, daemon=True).start()
    assert done.wait(timeout), "condition never became true"


def test_verdict_cached_for_matching_transcript():
    p = TurnPredictor(lambda text: _decision(CognitionAction.SPEAK))
    p.on_partial("so what do you think")
    _wait_for(lambda: p.verdict_for("so what do you think") is not None)
    v = p.verdict_for("So, what do you think?")  # punctuation/case-insensitive match
    assert v is not None
    assert v.action == CognitionAction.SPEAK


def test_diverged_transcript_returns_none():
    p = TurnPredictor(lambda text: _decision())
    p.on_partial("i was thinking about")
    _wait_for(lambda: p.verdict_for("i was thinking about") is not None)
    assert p.verdict_for("i was thinking about pizza actually") is None


def test_stale_judgment_discards_itself():
    release = threading.Event()
    calls = []

    def slow_decide(text):
        calls.append(text)
        if text == "first":
            release.wait(2.0)  # first judgment is slow...
            return _decision(CognitionAction.WAIT, "stale")
        return _decision(CognitionAction.SPEAK, "fresh")

    p = TurnPredictor(slow_decide)
    p.on_partial("first")
    p.on_partial("second")  # supersedes while the first is mid-flight
    _wait_for(lambda: p.verdict_for("second") is not None)
    release.set()  # let the stale judgment finish now
    _wait_for(lambda: len(calls) == 2)
    v = p.verdict_for("second")
    assert v is not None and v.thought == "fresh"  # stale result never overwrote
    assert p.verdict_for("first") is None


def test_unchanged_partial_does_not_rejudge():
    calls = []
    p = TurnPredictor(lambda text: (calls.append(text), _decision())[1])
    p.on_partial("hello there")
    _wait_for(lambda: p.verdict_for("hello there") is not None)
    p.on_partial("Hello, there!")  # same normalized basis
    assert calls == ["hello there"]


def test_reset_invalidates_verdict():
    p = TurnPredictor(lambda text: _decision())
    p.on_partial("all done here")
    _wait_for(lambda: p.verdict_for("all done here") is not None)
    p.reset()
    assert p.verdict_for("all done here") is None


def test_model_exception_leaves_no_verdict():
    def boom(text):
        raise RuntimeError("ollama down")

    p = TurnPredictor(boom)
    p.on_partial("hello there")
    import time

    time.sleep(0.05)
    assert p.verdict_for("hello there") is None  # net covers the pause; nothing crashes
