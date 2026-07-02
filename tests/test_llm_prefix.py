"""Tests for the LLM role-prefix stripper (llama3.2's leaked 'assistant' first token)."""

from orpheus_live.engines.llm import _strip_role_prefix


def test_strips_leaked_assistant_prefix():
    assert _strip_role_prefix("assistant So, I'm curious.") == "So, I'm curious."
    assert _strip_role_prefix("assistant: hey there") == "hey there"
    assert _strip_role_prefix("  assistant  hello") == "hello"


def test_strips_other_role_words():
    assert _strip_role_prefix("user what's up") == "what's up"
    assert _strip_role_prefix("system ready") == "ready"


def test_leaves_normal_replies_untouched():
    assert _strip_role_prefix("So, I'm curious about that.") == "So, I'm curious about that."
    # Only a leading *role word* is stripped, not a word that merely contains one.
    assert _strip_role_prefix("assistants are everywhere") == "assistants are everywhere"
    assert _strip_role_prefix("Assistant-like tools help.") == "Assistant-like tools help."
