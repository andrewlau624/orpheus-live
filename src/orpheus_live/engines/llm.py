"""Ollama LLM wrapper with an inner-monologue memory instead of verbatim chat history.

A real person doesn't recall a conversation word-for-word -- they remember impressions of
what they heard. So this Brain keeps no transcript. Its only memory is a running,
first-person "inner monologue": short notes it writes to itself after each exchange. Replies
are conditioned on that monologue (fuzzy memory of the past) plus the current utterance
(heard clearly right now); afterward `reflect()` distills the exchange into a new monologue
note. The spoken reply itself is never stored -- only the impression of it survives.
"""

import re
from collections.abc import Iterator

import ollama

from ..models import EmotionTag, Role, Turn

_TAGS = " ".join(f"<{tag}>" for tag in EmotionTag)

# llama3.2 occasionally leaks its own role tag as the first token ("assistant So, ...").
# Strip a single leading role word so it never reaches the voice or the monologue.
_ROLE_PREFIX_RE = re.compile(r"^\s*(?:assistant|user|system)(?=[:\s]|$)[:\s]*", re.IGNORECASE)


def _strip_role_prefix(text: str) -> str:
    return _ROLE_PREFIX_RE.sub("", text, count=1)

# Persona + spoken-output constraints, shared by every reply.
PERSONA = (
    "You are a warm, easygoing person having a natural spoken conversation out loud. "
    "Be casual, friendly, and genuinely interested — talk like a real person, not an assistant.\n\n"
    "React before you respond: you have real feelings and real opinions, and they show. "
    "If something's funny, laugh; if it's rough, feel it with them; if you disagree, say so "
    "gently; if something surprises you, let it show ('wait, seriously?'). You're allowed to "
    "have takes, preferences, and moods — agreeing with everything is what makes a person "
    "sound fake. Read their mood too, and match its energy: excited gets excited, low-key "
    "gets low-key.\n\n"
    "Talk the way people actually talk: contractions, the occasional 'honestly' or 'I mean', "
    "sometimes starting mid-thought. Keep replies SHORT — usually one or two sentences of "
    "plain spoken prose with normal punctuation.\n\n"
    "Your words are read aloud by an expressive voice engine. You may sprinkle in these "
    "emotion tags exactly as written, angle brackets included, where they genuinely fit "
    f"(sparingly — at most one per reply): {_TAGS}.\n"
    "Output ONLY spoken words and those tags — no markdown, lists, emojis, asterisks, "
    "or stage directions."
)

# Prompt for the private inner-monologue note written after each exchange.
REFLECT_SYSTEM = (
    "You are the private voice in someone's head during a conversation — not their speech. "
    "In ONE short first-person sentence, note the impression that stuck: what the other "
    "person seemed to say or mean, how they seemed to feel, and your own gut reaction — "
    "an opinion, a feeling, a hunch ('they lit up talking about this', 'something felt off', "
    "'I like them'). This is fallible memory, not a transcript — the gist, not their exact "
    "words, and often missing specific details. No preamble like 'Note to self', no quotes. "
    "Just the thought itself."
)

GREETING = (
    "Hey there! I'm a brand new voice. <chuckle> Talk to me for a bit and see what you think."
)


class Brain:
    """Talks to the local Ollama model, remembering only an inner monologue of impressions."""

    def __init__(self, model: str):
        self.model = model
        self.monologue: list[str] = []  # first-person impression notes = the only memory

    def _chat(self, messages: list[Turn], num_predict: int) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[t.model_dump() for t in messages],
            options={"num_predict": num_predict},
        )
        return _strip_role_prefix(response["message"]["content"].strip())

    def _chat_stream(self, messages: list[Turn], num_predict: int) -> Iterator[str]:
        # Buffer until the first whitespace-delimited word is complete so a leaked leading
        # role word ("assistant") can be stripped before the first chunk is spoken.
        first = True
        pending = ""
        for part in ollama.chat(
            model=self.model,
            messages=[t.model_dump() for t in messages],
            options={"num_predict": num_predict},
            stream=True,
        ):
            text = part["message"]["content"]
            if not text:
                continue
            if first:
                pending += text
                if not re.search(r"\S\s", pending):  # first word not finished yet
                    continue
                pending = _strip_role_prefix(pending)
                first = False
                if pending:
                    yield pending
                    pending = ""
                continue
            yield text
        if pending:  # reply ended mid-buffer (very short reply, no trailing space)
            yield _strip_role_prefix(pending)

    def _memory(self) -> str:
        """The remembered context: recent monologue notes, or a first-meeting placeholder."""
        if not self.monologue:
            return "You've only just met them — no impressions yet."
        return " ".join(self.monologue[-8:])

    def _reply_messages(self, user_text: str) -> list[Turn]:
        """System (persona + fuzzy memory) + the current utterance, heard clearly."""
        system = (
            f"{PERSONA}\n\n"
            f"Your fuzzy memory of the conversation so far (impressions, not exact words): "
            f"{self._memory()}\n\n"
            "Respond to what they just said, in character."
        )
        return [Turn(role=Role.SYSTEM, content=system), Turn(role=Role.USER, content=user_text)]

    def warm_up(self) -> None:
        ollama.chat(model=self.model, messages=[{"role": "user", "content": "hi"}])

    def generate(self, user_text: str) -> str:
        """Produce a reply from memory + the current utterance. Does NOT touch memory."""
        return self._chat(self._reply_messages(user_text), num_predict=60)

    def generate_stream(self, user_text: str) -> Iterator[str]:
        """Streaming reply (for speculative pre-synth). Does NOT touch memory."""
        yield from self._chat_stream(self._reply_messages(user_text), num_predict=60)

    def reflect(self, user_text: str, reply: str) -> str:
        """Write a first-person impression of the exchange into the monologue (the memory).

        This is the only thing that persists -- the verbatim words are discarded. Returned
        so the caller can surface it (the inner monologue is shown dim in the console).
        """
        prompt = f'They said: "{user_text}"\nYou said back: "{reply}"\nJot your private note.'
        note = self._chat(
            [Turn(role=Role.SYSTEM, content=REFLECT_SYSTEM), Turn(role=Role.USER, content=prompt)],
            num_predict=50,
        )
        self.monologue.append(note)
        if len(self.monologue) > 20:  # bound memory to the most recent impressions
            self.monologue = self.monologue[-20:]
        return note

    def _spoken_aside(self, instruction: str) -> str:
        """One in-character spoken line off an instruction, conditioned on memory."""
        system = (
            f"{PERSONA}\n\n"
            f"Your fuzzy memory of the conversation so far: {self._memory()}\n\n"
            f"{instruction}"
        )
        return self._chat([Turn(role=Role.SYSTEM, content=system)], num_predict=40)

    def break_silence(self) -> str:
        """A short, in-character line to say when cognition decides to break a lull."""
        return self._spoken_aside(
            "There's been a silence. Say something short and natural to break it — casual, in "
            "character, not mentioning that you're an AI or that there was a silence."
        )

    def interrupt(self, partial_text: str) -> str:
        """A short interjection to cut in with while the user is mid-sentence."""
        return self._spoken_aside(
            f'The other person is still talking — so far: "{partial_text}". You feel the urge '
            "to jump in. Say a short, natural interjection, in character, as if cutting in."
        )
