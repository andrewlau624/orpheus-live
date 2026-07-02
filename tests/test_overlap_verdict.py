"""Tests for the instant overlap-classifier heuristics (backchannel vs interrupt)."""

from orpheus_live.core.cognition import is_stop_command, quick_overlap_verdict


def test_empty_or_unintelligible_overlap_is_a_backchannel():
    assert quick_overlap_verdict("") == "backchannel"
    assert quick_overlap_verdict("...") == "backchannel"


def test_short_agreement_words_are_backchannels():
    assert quick_overlap_verdict("yeah") == "backchannel"
    assert quick_overlap_verdict("Oh wow.") == "backchannel"
    assert quick_overlap_verdict("mm right") == "backchannel"
    assert quick_overlap_verdict("no way!") == "backchannel"


def test_fillers_and_agreeing_murmurs_are_backchannels():
    # Common Whisper output for agreeing noises / short sounds while the AI talks.
    assert quick_overlap_verdict("mhm") == "backchannel"
    assert quick_overlap_verdict("uh huh") == "backchannel"
    assert quick_overlap_verdict("oh yeah totally") == "backchannel"
    # Up to four all-backchannel words still counts (agreeing run, not a turn-grab).
    assert quick_overlap_verdict("yeah yeah okay sure") == "backchannel"


def test_long_speech_is_always_an_interrupt():
    assert (
        quick_overlap_verdict("wait hold on I actually wanted to ask you about that thing")
        == "interrupt"
    )


def test_short_but_substantive_speech_goes_to_the_model():
    # not obviously a backchannel, not long enough to auto-interrupt -> gray zone
    assert quick_overlap_verdict("wait what about tomorrow") is None
    assert quick_overlap_verdict("can I ask something") is None


def test_explicit_stop_commands_interrupt_immediately():
    # Short "cut it out" commands must interrupt without a model consult, even though
    # they're too short for the >=8-word rule.
    assert quick_overlap_verdict("stop") == "interrupt"
    assert quick_overlap_verdict("stop talking please") == "interrupt"
    assert quick_overlap_verdict("hold on hold on") == "interrupt"
    assert is_stop_command("okay stop")
    assert is_stop_command("shut up")
    assert not is_stop_command("what about tomorrow")
    assert not is_stop_command("i wanted to ask")
