"""Tests for AudioIn's continuous, pause-triggered segmentation (VAD as a speech gate only)."""

import threading

from conftest import FakeVad
from orpheus_live.audio.capture import AudioIn
from orpheus_live.config import Settings

_FRAME = b"\x00\x00" * 16


def _make_audio_in(vad, on_pause=None, muted=None):
    settings = Settings(
        frame_ms=32,
        start_speech_ms=64,  # 2 voiced frames -> turn starts
        min_utterance_ms=32,  # 1 frame banked before a pause is judged
        turn_pause_ms=96,  # 3 silent frames -> short pause (consult)
        turn_end_ms=224,  # 7 silent frames -> forced turn end
        post_speak_cooldown=0.0,
    )
    speaking = threading.Event()
    audio_in = AudioIn(
        settings,
        vad,
        speaking,
        [0.0],
        on_pause=on_pause,
        muted=muted,
        clock=lambda: 1000.0,  # always well past cooldown
    )
    return audio_in, speaking


def test_pause_fires_once_after_speech_then_silence():
    events = []
    # 3 voiced (turn starts at 2), then 3 silent -> one short-pause event.
    vad = FakeVad([True, True, True, False, False, False])
    audio_in, _ = _make_audio_in(vad, on_pause=lambda a, final: events.append(final))

    for _ in range(6):
        audio_in._process_frame(_FRAME)

    assert events == [False]  # exactly one non-final pause event


def test_resumed_speech_rearms_the_pause():
    events = []
    # speak, short pause (fires), speak again, short pause (fires again)
    script = [True, True, True, False, False, False, True, True, False, False, False]
    vad = FakeVad(script)
    audio_in, _ = _make_audio_in(vad, on_pause=lambda a, final: events.append(final))

    for _ in range(len(script)):
        audio_in._process_frame(_FRAME)

    assert events == [False, False]  # a fresh burst re-arms the short-pause trigger


def test_resumed_speech_rearms_the_final_safety_net():
    # Regression: multiple sub-utterances in one turn. Each long silence must be able to
    # fire its own final=True, or a turn hangs forever when cognition keeps saying "wait".
    events = []
    # speak -> 7 silent (pause + final) -> speak again -> 7 silent (pause + final AGAIN)
    script = [True, True] + [False] * 7 + [True, True] + [False] * 7
    vad = FakeVad(script)
    audio_in, _ = _make_audio_in(vad, on_pause=lambda a, final: events.append(final))

    for _ in range(len(script)):
        audio_in._process_frame(_FRAME)

    # Two full cycles: the final safety net must fire in BOTH, not just the first.
    assert events.count(True) == 2
    assert events == [False, True, False, True]


def test_long_silence_fires_final_turn_end():
    events = []
    # 2 voiced -> trigger, then 7 silent -> short pause AND the forced final end.
    vad = FakeVad([True, True] + [False] * 7)
    audio_in, _ = _make_audio_in(vad, on_pause=lambda a, final: events.append(final))

    for _ in range(9):
        audio_in._process_frame(_FRAME)

    assert False in events and True in events  # both the pause and the final safety-net


def test_no_pause_without_speech_onset():
    events = []
    vad = FakeVad([False] * 10)  # never enough voiced frames to start a turn
    audio_in, _ = _make_audio_in(vad, on_pause=lambda a, final: events.append(final))

    for _ in range(10):
        audio_in._process_frame(_FRAME)

    assert events == []


def test_reset_turn_clears_state():
    vad = FakeVad([True, True, True])
    audio_in, _ = _make_audio_in(vad, on_pause=lambda a, final: None)

    for _ in range(3):
        audio_in._process_frame(_FRAME)
    assert audio_in.turn_audio() is not None  # a turn is active

    audio_in.reset_turn()
    assert audio_in.turn_audio() is None  # cleared


def test_muted_mic_fires_no_pause_and_drops_the_turn():
    # Lag-aware pickup: while muted, the mic is ignored -- no turn forms, no pause fires,
    # even on clearly-voiced frames (the "thinking" gap before the AI's audio is out).
    events = []
    muted = threading.Event()
    muted.set()
    vad = FakeVad([True] * 6)
    audio_in, _ = _make_audio_in(vad, on_pause=lambda a, final: events.append(final), muted=muted)

    for _ in range(6):
        audio_in._process_frame(_FRAME)

    assert events == []
    assert audio_in.turn_audio() is None  # nothing accumulated while muted


def test_unmuting_resumes_normal_pickup():
    events = []
    muted = threading.Event()
    muted.set()
    # Muted frames never reach the VAD, so the script covers only the 5 post-unmute frames:
    # 2 voiced -> trigger, then 3 silent -> one short pause.
    vad = FakeVad([True, True, False, False, False])
    audio_in, _ = _make_audio_in(vad, on_pause=lambda a, final: events.append(final), muted=muted)

    for i in range(8):  # 3 muted (ignored) frames, then 5 normal ones
        if i == 3:
            muted.clear()  # first audio became audible -> mic resumes
        audio_in._process_frame(_FRAME)

    assert events == [False]  # a normal turn forms and pauses once after unmuting


def test_captures_while_ai_speaking():
    """Continuous capture: a turn still forms while the AI is talking (interruptions heard)."""
    events = []
    vad = FakeVad([True, True, True, False, False, False])
    audio_in, speaking = _make_audio_in(vad, on_pause=lambda a, final: events.append(final))
    speaking.set()  # AI is talking

    for _ in range(6):
        audio_in._process_frame(_FRAME)

    assert events == [False]  # the pause still fires so cognition can judge the interrupt
    assert audio_in.turn_audio() is not None
