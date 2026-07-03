"""Tests for the stop-command reflex — the one non-model shortcut kept on purpose.

Everything else about turn-taking and overlap judgment is model-driven (see
TurnPredictor and decide_turn); "stop" must halt playback instantly, reflex-fast.
"""

from orpheus_live.core.cognition import is_stop_command


def test_explicit_stop_commands_are_recognized():
    assert is_stop_command("stop")
    assert is_stop_command("okay stop")
    assert is_stop_command("stop talking please")
    assert is_stop_command("shut up")
    assert is_stop_command("hold on hold on")
    assert is_stop_command("Wait wait!")


def test_ordinary_speech_is_not_a_stop_command():
    assert not is_stop_command("what about tomorrow")
    assert not is_stop_command("i wanted to ask")
    assert not is_stop_command("keep going this is great")


# -- text-domain echo rejection --------------------------------------------------

from orpheus_live.core.cognition import looks_like_echo  # noqa: E402


def test_echo_of_own_words_is_detected():
    spoken = "I'm really into hiking these days, it's been great for clearing my head"
    assert looks_like_echo("into hiking these days", spoken)
    assert looks_like_echo("clearing my head", spoken)


def test_genuine_new_turn_is_not_echo():
    spoken = "I'm really into hiking these days, it's been great"
    assert not looks_like_echo("wait what about tomorrow night", spoken)
    assert not looks_like_echo("no I totally disagree with that", spoken)


def test_empty_overlap_over_speech_counts_as_echo_noise():
    assert looks_like_echo("", "anything the ai is saying")
    assert looks_like_echo("...", "anything the ai is saying")


def test_no_spoken_text_is_never_echo():
    assert not looks_like_echo("hello there", "")
