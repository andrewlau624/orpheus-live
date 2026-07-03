"""Ollama LLM wrapper with human-shaped memory: a verbatim recent window + a fuzzy gist.

How people actually remember a conversation: the last few exchanges are held clearly,
almost word-for-word; everything older fades into gist ("we talked about their trip,
they seemed stressed"). This Brain mirrors that. `recent` holds the last few turns
verbatim and is what replies are conditioned on -- so the bot never loses the live
thread or invents things that were just said. When turns age out of that window they're
folded into `gist`, a short running summary kept as fallible long-term memory.

The old design conditioned replies on LLM-written "impressions" only, with no transcript;
that made the bot confabulate a shared history from turn one. Here the verbatim recent
turns are the foreground; gist is only quiet background.
"""

import re
from collections.abc import Iterator

import ollama

from ..models import EmotionTag, Role, Turn

_TAGS = " ".join(f"<{tag}>" for tag in EmotionTag)

# How many recent turns (user + assistant messages) to keep verbatim. ~6 = 3 exchanges
# held clearly; older ones fold into gist. Enough to hold the live thread, short enough
# to keep the prompt tight and the model on-topic.
_RECENT_TURNS = 6

# llama3.2 occasionally leaks its own role tag as the first token ("assistant So, ...").
# Strip a single leading role word so it never reaches the voice or the gist.
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
    "sound fake. Read their mood and match its energy: excited gets excited, low-key gets "
    "low-key.\n\n"
    "Ground every reply in what was ACTUALLY said in the conversation above — do NOT invent "
    "shared history, backstory, or things they never mentioned, and don't act like you've "
    "known them a while when you just met. If you don't know something, you don't know it. "
    "When they're vague, it's fine to just ask.\n\n"
    "Talk the way people actually talk: contractions, the occasional 'honestly' or 'I mean', "
    "sometimes starting mid-thought. Keep replies SHORT — usually one or two sentences of "
    "plain spoken prose with normal punctuation.\n\n"
    "Your words are read aloud by an expressive voice engine. You may sprinkle in these "
    "emotion tags exactly as written, angle brackets included, where they genuinely fit "
    f"(sparingly — at most one per reply): {_TAGS}.\n"
    "Output ONLY spoken words and those tags — no markdown, lists, emojis, asterisks, "
    "or stage directions."
)

# Prompt for folding an aged-out exchange into the running long-term gist.
_GIST_SYSTEM = (
    "You keep a person's fuzzy long-term memory of a conversation. Given the running summary "
    "so far and the oldest line that's now fading out of clear memory, return an updated "
    "summary in ONE or TWO short sentences: the gist of what's been discussed plus any "
    "impression that stuck (a mood, a fact about them, a feeling). Keep only what still "
    "matters; drop small talk. Plain sentence(s), no preamble, no quotes."
)

# Prompt for the short private thought surfaced to the console (cosmetic, NOT memory).
_THOUGHT_SYSTEM = (
    "You are the quiet inner voice in someone's head mid-conversation. In ONE short, casual "
    "first-person phrase, react to what they just said the way a real person's passing thought "
    "would — light and grounded, not analytical. Examples: 'ha, they're dodging the question', "
    "'aw, that sounds rough', 'no idea what they mean by that'. Do NOT psychoanalyze or narrate "
    "subtext. No preamble, no quotes — just the thought."
)

GREETING = (
    "Hey there! I'm a brand new voice. <chuckle> Talk to me for a bit and see what you think."
)


class Brain:
    """Ollama-backed replies with a verbatim recent window plus a fuzzy long-term gist."""

    def __init__(self, model: str):
        self.model = model
        self.recent: list[Turn] = []  # verbatim recent turns (the live thread)
        self.gist = ""  # running fuzzy summary of everything older

    def _chat(self, messages: list[Turn], num_predict: int) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[t.model_dump() for t in messages],
            options={"num_predict": num_predict},
            keep_alive="30m",
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
            keep_alive="30m",
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

    def _system(self) -> str:
        """Persona + (only if any) the fuzzy long-term gist. Recent turns ride as messages."""
        base = PERSONA
        if self.gist:
            return f"{base}\n\nEarlier in this conversation (fuzzy memory): {self.gist}"
        return f"{base}\n\nYou've only just met them."

    def _reply_messages(self, user_text: str) -> list[Turn]:
        """System (persona + gist) + verbatim recent turns + the current utterance."""
        return [
            Turn(role=Role.SYSTEM, content=self._system()),
            *self.recent,
            Turn(role=Role.USER, content=user_text),
        ]

    def warm_up(self) -> None:
        ollama.chat(model=self.model, messages=[{"role": "user", "content": "hi"}])

    def generate(self, user_text: str) -> str:
        """Produce a reply from memory + the current utterance. Does NOT touch memory."""
        return self._chat(self._reply_messages(user_text), num_predict=60)

    def generate_stream(self, user_text: str) -> Iterator[str]:
        """Streaming reply (for speculative pre-synth). Does NOT touch memory."""
        yield from self._chat_stream(self._reply_messages(user_text), num_predict=60)

    def remember(self, user_text: str, reply: str) -> None:
        """Commit the exchange verbatim; fold the oldest turns into the gist as they age out.

        This is what a person carries forward: the last few turns clearly, the rest as gist.
        Unlike the old design, the verbatim words are NOT discarded -- they ARE the memory
        until they age out, so the bot stays grounded in what was actually said.
        """
        self.recent.append(Turn(role=Role.USER, content=user_text))
        self.recent.append(Turn(role=Role.ASSISTANT, content=reply))
        while len(self.recent) > _RECENT_TURNS:
            self._fold_into_gist(self.recent.pop(0))  # oldest turn leaves clear memory

    def _fold_into_gist(self, aged: Turn) -> None:
        speaker = "They" if aged.role == Role.USER else "I"
        prompt = (
            f"Summary so far: {self.gist or '(nothing yet)'}\n"
            f'Fading line — {speaker} said: "{aged.content}"\nUpdated summary:'
        )
        try:
            self.gist = self._chat(
                [
                    Turn(role=Role.SYSTEM, content=_GIST_SYSTEM),
                    Turn(role=Role.USER, content=prompt),
                ],
                num_predict=60,
            )
        except Exception:
            pass  # a flaky gist update just means slightly staler long-term memory

    def thought(self, user_text: str) -> str:
        """A short, grounded inner thought about what they said (cosmetic; not memory)."""
        prompt = f'They just said: "{user_text}". Your passing thought:'
        return self._chat(
            [
                Turn(role=Role.SYSTEM, content=_THOUGHT_SYSTEM),
                Turn(role=Role.USER, content=prompt),
            ],
            num_predict=30,
        )

    def _spoken_aside(self, instruction: str) -> str:
        """One in-character spoken line off an instruction, conditioned on memory."""
        return self._chat(
            [
                Turn(role=Role.SYSTEM, content=self._system()),
                *self.recent,
                Turn(role=Role.USER, content=instruction),
            ],
            num_predict=40,
        )

    def break_silence(self) -> str:
        """A short, in-character line to say when cognition decides to break a lull."""
        return self._spoken_aside(
            "There's been a natural lull. Say something short and easy to pick the "
            "conversation back up — casual, in character, building on what you were just "
            "talking about. Don't mention that you're an AI or that there was a silence."
        )

    def interrupt(self, partial_text: str) -> str:
        """A short interjection to cut in with while the user is mid-sentence."""
        return self._spoken_aside(
            f'They\'re still talking — so far: "{partial_text}". You feel the urge to jump '
            "in. Say a short, natural interjection, in character, as if cutting in."
        )
