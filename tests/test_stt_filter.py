"""Tests for the STT repetition-hallucination filter (the 'rafa rafa rafa' guard)."""

from orpheus_live.engines.stt import _looks_like_loop


def test_repeated_token_loop_is_flagged():
    assert _looks_like_loop("rafa rafa rafa rafa rafa rafa rafa")
    assert _looks_like_loop("yeah yeah yeah yeah yeah yeah")


def test_normal_speech_is_not_flagged():
    assert not _looks_like_loop("hey, what are you up to this weekend?")
    assert not _looks_like_loop("I was thinking we could grab lunch tomorrow")


def test_short_phrases_pass_even_with_repetition():
    # Too short to confidently call a loop; real speech like "no no no" must survive.
    assert not _looks_like_loop("no no no")
    assert not _looks_like_loop("yeah")


def test_genuine_repeated_greeting_survives():
    # Regression: saying "hello" a few times is real speech, not a "rafa rafa" loop.
    assert not _looks_like_loop("hello hello hello")
    assert not _looks_like_loop("hello hello hello hello")
    assert not _looks_like_loop("hi hi hi hi hi")


def test_mostly_repeated_with_a_little_filler_still_flagged():
    assert _looks_like_loop("um rafa rafa rafa rafa rafa rafa")
