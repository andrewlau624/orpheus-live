"""Tests for Orpheus input hygiene: keep known <tags> and prose, drop the rest."""

from orpheus_live.engines.sanitize import clean_for_tts, strip_markers


def test_keeps_allowed_tags():
    assert clean_for_tts("Hi there. <laugh> Good to see you. <sigh>") == (
        "Hi there. <laugh> Good to see you. <sigh>"
    )


def test_drops_unknown_tags_and_bracketed_stage_directions():
    out = clean_for_tts("Hi <sparkle> there [stage direction] friend (aside). <laugh>")
    assert "<sparkle>" not in out
    assert "stage" not in out
    assert "aside" not in out
    assert "<laugh>" in out


def test_normal_punctuation_survives():
    # Orpheus is Llama-based -- apostrophes, questions, exclamations are fine.
    assert clean_for_tts("It's a no-brainer, isn't it?!") == "It's a no-brainer, isn't it?!"


def test_markdown_and_emojis_are_stripped():
    out = clean_for_tts("**Sure!** Here's a list: • one \U0001f600")
    assert "*" not in out
    assert "•" not in out
    assert "\U0001f600" not in out
    assert "Sure!" in out


def test_curly_quotes_normalized():
    assert clean_for_tts("it’s “fine”") == "it's fine"


def test_empty_input_becomes_hmm():
    assert clean_for_tts("") == "Hmm."
    assert clean_for_tts("\U0001f600\U0001f600") == "Hmm."


def test_no_space_before_punctuation():
    assert clean_for_tts("thanks , see you !") == "thanks, see you!"


def test_strip_markers_for_console_display():
    assert strip_markers("Hey there. <laugh> Good stuff.") == "Hey there. Good stuff."
