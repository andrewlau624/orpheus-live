"""Conversation-level types: message turns and the live turn-taking state."""

from enum import StrEnum

from pydantic import BaseModel


class Role(StrEnum):
    """Chat roles, as Ollama expects them on the wire."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Turn(BaseModel):
    """One message in the conversation history."""

    role: Role
    content: str


class ConversationState(StrEnum):
    """Where the live loop currently is — the single field every loop reads/writes."""

    IDLE = "idle"  # mutual silence, waiting
    LISTENING = "listening"  # user is mid-utterance
    PREPARING = "preparing"  # user stopped; generating the reply
    SPEAKING = "speaking"  # AI is talking
    OVERLAP = "overlap"  # user started talking over the AI
