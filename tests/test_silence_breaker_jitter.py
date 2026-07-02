from conftest import FakeClock
from orpheus_live.config import Settings
from orpheus_live.core.cognition import SilenceCognition
from orpheus_live.models import CognitionDecision


def _scripted_consult(decisions):
    calls = []

    def consult(silence_s, consult_count):
        calls.append((silence_s, consult_count))
        idx = min(len(calls) - 1, len(decisions) - 1)
        return decisions[idx]

    consult.calls = calls
    return consult


def test_eventually_speaks_after_scripted_wait_wait_speak():
    clock = FakeClock()
    decisions = [
        CognitionDecision(action="wait", urgency=0.1, thought="should I? nah"),
        CognitionDecision(action="wait", urgency=0.3, thought="still nothing... hold on"),
        CognitionDecision(action="speak", urgency=0.8, thought="ok, breaking it"),
    ]
    consult = _scripted_consult(decisions)
    settings = Settings(cognition_base_silence_s=2.0, cognition_jitter_frac=0.0)
    cognition = SilenceCognition(settings, consult=consult, clock=clock.now, rng=lambda a, b: 0.0)

    cognition.note_silence_start()
    results = []
    for _ in range(30):
        clock.advance(0.4)
        decision = cognition.tick()
        if decision is not None:
            results.append(decision)
            if decision.action == "speak":
                break

    assert [r.action for r in results] == ["wait", "wait", "speak"]


def test_check_interval_is_jittered_not_fixed():
    clock = FakeClock()
    decisions = [CognitionDecision(action="wait", urgency=0.1, thought="t")] * 5
    consult = _scripted_consult(decisions)
    settings = Settings(cognition_base_silence_s=2.0, cognition_jitter_frac=0.4)

    jitters = iter([0.5, -0.3, 0.2, -0.6, 0.1])
    cognition = SilenceCognition(
        settings, consult=consult, clock=clock.now, rng=lambda a, b: next(jitters)
    )
    cognition.note_silence_start()

    intervals = []
    last_check = clock.now()
    for _ in range(60):
        clock.advance(0.1)
        decision = cognition.tick()
        if decision is not None:
            intervals.append(round(clock.now() - last_check, 2))
            last_check = clock.now()
        if len(intervals) >= 3:
            break

    assert len(set(intervals)) > 1


def test_reset_clears_tracking_and_restarts_from_scratch():
    clock = FakeClock()
    consult = _scripted_consult([CognitionDecision(action="speak", urgency=1.0, thought="t")])
    settings = Settings(cognition_base_silence_s=1.0, cognition_jitter_frac=0.0)
    cognition = SilenceCognition(settings, consult=consult, clock=clock.now, rng=lambda a, b: 0.0)

    cognition.note_silence_start()
    clock.advance(0.5)
    cognition.reset()
    clock.advance(0.6)
    assert cognition.tick() is None  # silence tracking was reset, no check due yet

    cognition.note_silence_start()
    clock.advance(1.0)
    decision = cognition.tick()
    assert decision is not None and decision.action == "speak"
