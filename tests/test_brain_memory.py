"""Tests for Brain's human-shaped memory: verbatim recent window + folded gist."""

from orpheus_live.engines import llm
from orpheus_live.engines.llm import Brain
from orpheus_live.models import Role


def _brain(monkeypatch, capture=None):
    """A Brain whose ollama calls are faked; optionally capture the messages sent."""

    def fake_chat(**kwargs):
        if capture is not None:
            capture.append(kwargs.get("messages", []))
        return {"message": {"content": "sure, sounds good"}}

    monkeypatch.setattr(llm.ollama, "chat", fake_chat)
    return Brain("fake-model")


def test_recent_turns_are_kept_verbatim_and_sent_to_the_model(monkeypatch):
    sent = []
    b = _brain(monkeypatch, sent)
    b.remember("I love pizza", "nice, what kind")
    b.generate("cheese, obviously")
    msgs = sent[-1]
    contents = [m["content"] for m in msgs]
    assert "I love pizza" in contents  # actual words, not an impression
    assert "nice, what kind" in contents
    assert msgs[-1] == {"role": Role.USER.value, "content": "cheese, obviously"}


def test_recent_window_is_bounded_and_older_turns_fold_into_gist(monkeypatch):
    b = _brain(monkeypatch)
    for i in range(6):
        b.remember(f"user says {i}", f"reply {i}")
    assert len(b.recent) <= llm._RECENT_TURNS
    assert b.gist  # older exchanges were summarized into the running gist
    # The freshest exchange is still held verbatim.
    assert any("user says 5" == t.content for t in b.recent)


def test_no_gist_before_anything_ages_out(monkeypatch):
    b = _brain(monkeypatch)
    b.remember("hi", "hey there")
    assert b.gist == ""  # nothing has aged out yet -> no fuzzy long-term memory yet


def test_system_prompt_says_just_met_when_empty(monkeypatch):
    b = _brain(monkeypatch)
    assert "only just met" in b._system()
