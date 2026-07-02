"""Voice identity: the eight Orpheus base voices, emotion tags, and named presets."""

from enum import StrEnum

from pydantic import BaseModel, Field


class Voice(StrEnum):
    """Orpheus's eight built-in base voices (it has no speaker embeddings)."""

    TARA = "tara"
    LEAH = "leah"
    JESS = "jess"
    LEO = "leo"
    DAN = "dan"
    MIA = "mia"
    ZAC = "zac"
    ZOE = "zoe"


class EmotionTag(StrEnum):
    """The paralinguistic tags Orpheus renders; anything else is stripped before TTS.

    Single source of truth for the tag set — the LLM system prompt, the sanitizer's
    allow-list, and the sentence splitter all derive from this.
    """

    LAUGH = "laugh"
    CHUCKLE = "chuckle"
    SIGH = "sigh"
    COUGH = "cough"
    SNIFFLE = "sniffle"
    GROAN = "groan"
    YAWN = "yawn"
    GASP = "gasp"


# Back-compat tuple for callers/tests that want a plain iterable of voice strings.
ORPHEUS_VOICES: tuple[Voice, ...] = tuple(Voice)


class VoicePreset(BaseModel):
    """One nameable Orpheus voice: a base voice plus delivery parameters.

    Presets are what `saved_voices/<Name>.json` stores and what `orpheus-live
    <name>` loads. `repetition_penalty >= 1.1` is required by Orpheus for
    stable generations.
    """

    name: str = Field(min_length=1)
    voice: Voice
    temperature: float = Field(gt=0, le=1.5)
    top_p: float = Field(gt=0, le=1)
    repetition_penalty: float = Field(ge=1.1, le=2.0)
