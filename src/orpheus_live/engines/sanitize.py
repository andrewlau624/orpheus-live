"""Turn LLM replies into text Orpheus can speak cleanly.

Orpheus rides a Llama-3 tokenizer, so normal prose -- apostrophes, hyphens,
question marks, exclamations -- is all fine. Cleaning here is just hygiene:
keep the eight paralinguistic <tags> Orpheus understands, drop anything else
the LLM invents (markdown, emojis, stage directions, unknown tags).
"""

import re

from ..models import EmotionTag

ALLOWED_TAGS = frozenset(EmotionTag)
_TAG_RE = re.compile(r"<([a-z_]+)>")
_BRACKET_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")  # [stage direction] / (aside)


def clean_for_tts(text: str) -> str:
    """Reduce an LLM reply to speakable prose plus Orpheus's known <tags>."""
    # 1. protect allowed <tags> as placeholders; drop unknown <...> tokens
    kept: list[str] = []

    def _stash(m: re.Match) -> str:
        if m.group(1) in ALLOWED_TAGS:
            kept.append(m.group(1))
            return f"\x00{len(kept) - 1}\x00"
        return " "

    text = _TAG_RE.sub(_stash, text)

    # 2. drop bracketed/parenthesized stage directions and normalize quotes
    text = _BRACKET_RE.sub(" ", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", "").replace("”", "")
    text = text.replace("&", " and ")

    # 3. whitelist plain spoken characters (kills emojis, markdown, stray symbols)
    text = re.sub(r"[^a-zA-Z0-9 .,!?'\-\x00]", " ", text)

    # 4. restore tags, tidy spacing/punctuation
    text = re.sub(r"\x00(\d+)\x00", lambda m: f"<{kept[int(m.group(1))]}>", text)
    text = re.sub(r"\s+([.,!?])", r"\1", text)  # no space before punctuation
    text = re.sub(r"\s+", " ", text).strip()

    return text or "Hmm."


def strip_markers(text: str) -> str:
    """Human-readable version (tags removed) for the console transcript."""
    t = _TAG_RE.sub(" ", text)
    t = re.sub(r"\s+([.,?!])", r"\1", re.sub(r"\s+", " ", t))
    return t.strip()
