"""Cognition types: the actions the inner-monologue model can choose, and its decision."""

from enum import StrEnum

from pydantic import BaseModel, Field


class CognitionAction(StrEnum):
    """Actions the cognition model may return across its several prompts."""

    SPEAK = "speak"  # break a silence
    WAIT = "wait"  # hold / keep listening
    BACKCHANNEL = "backchannel"  # overlap was just an adlib — keep talking
    INTERRUPT = "interrupt"  # a real turn-grab — yield


class CognitionDecision(BaseModel):
    """A structured inner-monologue decision from the cognition LLM."""

    action: CognitionAction
    urgency: float = Field(ge=0, le=1)
    thought: str
