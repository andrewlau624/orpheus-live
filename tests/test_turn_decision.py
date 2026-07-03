"""Tests for reply-latency-critical decision paths.

Covers the mid-decision safety-net upgrade (WAIT -> SPEAK when final fired while the
model was thinking) and the Ollama latency knobs (keep_alive pinning, temperature 0).
"""

import numpy as np
import pytest

from orpheus_live.config import Settings
from orpheus_live.core import cognition as cognition_mod
from orpheus_live.core import conversation as conv_mod
from orpheus_live.models import CognitionAction, CognitionDecision

# -- ollama call knobs ---------------------------------------------------------


def _fake_chat_capture(captured):
    def fake_chat(**kwargs):
        captured.update(kwargs)
        return {
            "message": {
                "content": CognitionDecision(
                    action=CognitionAction.WAIT, urgency=0.1, thought="hmm"
                ).model_dump_json()
            }
        }

    return fake_chat


def test_decide_turn_pins_model_and_decodes_deterministically(monkeypatch):
    captured = {}
    monkeypatch.setattr(cognition_mod.ollama, "chat", _fake_chat_capture(captured))
    cognition_mod.decide_turn("m", "hello there", ai_speaking=False)
    assert captured["keep_alive"] == "30m"  # no model reload on the critical path
    assert captured["options"]["temperature"] == 0


def test_consult_pins_model_and_decodes_deterministically(monkeypatch):
    captured = {}
    monkeypatch.setattr(cognition_mod.ollama, "chat", _fake_chat_capture(captured))
    cognition_mod.consult("m", 3.0, 1)
    assert captured["keep_alive"] == "30m"
    assert captured["options"]["temperature"] == 0


# -- mid-decision safety-net upgrade -------------------------------------------


class _FakeTranscriber:
    def __init__(self, settings):
        pass

    def transcribe(self, audio):
        return "hello there my friend"

    def warm_up(self):
        pass


class _FakeBrain:
    def __init__(self, model):
        pass

    def generate_stream(self, text):
        yield "ok."

    def warm_up(self):
        pass


class _FakeVoice:
    preset = None

    def stream(self, text):
        yield np.zeros(10, dtype=np.float32)


class _NoSink:
    def __init__(self, settings):
        pass

    def begin(self, epoch, on_audible=None):
        pass

    def pace(self, expected_s):
        pass

    def write(self, chunk, epoch):
        pass

    def flush(self, epoch):
        pass

    def clear(self):
        pass

    def close(self):
        pass


@pytest.fixture
def conv(monkeypatch):
    monkeypatch.setattr(conv_mod, "load_voice", lambda settings, preset: _FakeVoice())
    monkeypatch.setattr(conv_mod, "Transcriber", _FakeTranscriber)
    monkeypatch.setattr(conv_mod, "Brain", _FakeBrain)
    monkeypatch.setattr(conv_mod, "Vad", lambda threshold, sr: object())
    monkeypatch.setattr(conv_mod, "AudioSink", _NoSink)
    c = conv_mod.Conversation(Settings())
    c._responded = []
    monkeypatch.setattr(c, "_respond", lambda text: c._responded.append(text))
    return c


def _decide_sync(conv, *, final=False, ai_speaking=False):
    """Run _decide the way _on_pause would: with the single-flight lock held."""
    conv._deciding.acquire()
    conv._decide(np.zeros(16000, dtype=np.float32), final, ai_speaking)


def test_missed_final_upgrades_wait_to_speak_without_second_pass(conv, monkeypatch):
    monkeypatch.setattr(
        conv_mod,
        "decide_turn",
        lambda model, text, ai_speaking: CognitionDecision(
            action=CognitionAction.WAIT, urgency=0.1, thought="mid-thought?"
        ),
    )
    # The 900ms safety net fired while the model was deliberating and was dropped
    # by the single-flight gate:
    conv._decision_missed_final = True
    _decide_sync(conv, final=False, ai_speaking=False)
    assert conv._responded == ["hello there my friend"]  # spoke NOW, no second pass
    assert conv._decision_missed_final is False  # net consumed, not re-fired


def test_wait_stays_wait_without_missed_final(conv, monkeypatch):
    monkeypatch.setattr(
        conv_mod,
        "decide_turn",
        lambda model, text, ai_speaking: CognitionDecision(
            action=CognitionAction.WAIT, urgency=0.1, thought="mid-thought?"
        ),
    )
    _decide_sync(conv, final=False, ai_speaking=False)
    assert conv._responded == []  # genuine WAIT: keep listening


def test_no_upgrade_while_ai_is_speaking(conv, monkeypatch):
    monkeypatch.setattr(
        conv_mod,
        "decide_turn",
        lambda model, text, ai_speaking: CognitionDecision(
            action=CognitionAction.WAIT, urgency=0.1, thought="holding the floor"
        ),
    )
    conv._decision_missed_final = True
    conv.speaking.set()  # AI holds the floor; WAIT means keep talking, not reply
    _decide_sync(conv, final=False, ai_speaking=True)
    assert conv._responded == []


def test_lock_is_released_even_when_decision_raises(conv, monkeypatch):
    def boom(model, text, ai_speaking):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(conv_mod, "decide_turn", boom)
    _decide_sync(conv, final=False, ai_speaking=False)
    assert not conv._deciding.locked()  # single-flight gate never wedges


# -- model verdicts pre-computed during speech drive the pause ------------------


class _StubPredictor:
    def __init__(self, verdict=None):
        self.verdict = verdict
        self.kicked = []

    def verdict_for(self, text):
        return self.verdict

    def on_partial(self, text):
        self.kicked.append(text)

    def reset(self):
        self.verdict = None


def test_cached_speak_verdict_responds_at_the_short_pause(conv):
    conv.turn_predictor = _StubPredictor(
        CognitionDecision(action=CognitionAction.SPEAK, urgency=0.8, thought="they're done")
    )
    _decide_sync(conv, final=False, ai_speaking=False)
    assert conv._responded == ["hello there my friend"]  # spoke at 500ms, not the 900ms net


def test_cached_wait_verdict_is_honored_no_heuristic_override(conv, monkeypatch):
    # Punctuated, "complete-looking" text: the old heuristic would have SPOKEN here.
    # The model said they're mid-thought -> model authority wins.
    monkeypatch.setattr(conv.transcriber, "transcribe", lambda audio: "I went to the store.")
    conv.turn_predictor = _StubPredictor(
        CognitionDecision(action=CognitionAction.WAIT, urgency=0.2, thought="mid-story")
    )
    _decide_sync(conv, final=False, ai_speaking=False)
    assert conv._responded == []


def test_no_verdict_waits_and_kicks_a_judgment_for_this_text(conv):
    stub = _StubPredictor(None)
    conv.turn_predictor = stub
    _decide_sync(conv, final=False, ai_speaking=False)
    assert conv._responded == []  # no verdict yet -> wait for the net
    assert stub.kicked == ["hello there my friend"]  # judgment now in flight


def test_final_net_speaks_regardless_of_missing_verdict(conv):
    conv.turn_predictor = _StubPredictor(None)
    _decide_sync(conv, final=True, ai_speaking=False)
    assert conv._responded == ["hello there my friend"]  # liveness: never hang a turn
