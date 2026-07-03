"""Silence self-questioning: "should I talk? ... nah, maybe I should break it".

A background tick that, once mutual silence lingers past a jittered base
threshold, consults a small/fast dedicated Ollama model for a structured
inner-monologue decision. The `thought` is always surfaced to the console,
whatever the action, so the hesitation is visible even when the AI stays
quiet -- this is deliberately prompt-driven (not a hardcoded probability
curve) so the pacing reads as human hesitation rather than a timer.
"""

import random
import re
import time
from collections.abc import Callable

import ollama

from ..config import Settings
from ..console import DIM, log
from ..models import CognitionAction, CognitionDecision

COGNITION_SYSTEM_PROMPT = (
    "You are the inner monologue of one participant in a live spoken "
    "conversation, during a lull where nobody is talking. You're a real "
    "person sitting with someone in silence -- and silence FEELS like "
    "something: comfortable, awkward, heavy, companionable. Decide whether "
    "to break it.\n\n"
    "The longer the quiet stretches, the more natural it is to lean toward "
    "speaking -- like your own hesitation fading. Don't always wait, and "
    "don't jump in instantly either -- vary it.\n\n"
    "Your thought is a real feeling about THIS moment with THIS person -- "
    "restlessness, curiosity about what they're thinking, contentment that "
    "the quiet is nice, mild worry you said something odd. Never think about "
    "timers, checks, silences-as-a-task, or the fact that you're deciding -- "
    "just be the person in the room. React like a person, not a machine.\n\n"
    "Respond ONLY as JSON matching the schema. `thought` is a short, "
    "first-person inner monologue with genuine emotion in it. "
    '`action` is "speak" or "wait" ("backchannel"/"interrupt" aren\'t '
    "available yet). `urgency` is 0..1, how strongly you feel about it."
)


def consult(model: str, silence_s: float, consult_count: int) -> CognitionDecision:
    """Ask the dedicated cognition model for a structured silence decision."""
    mood = (
        "just a beat" if silence_s < 3 else ("a while now" if silence_s < 8 else "a long stretch")
    )
    prompt = f"The quiet has gone on for {mood}."
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": COGNITION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        format=CognitionDecision.model_json_schema(),
        options={"num_predict": 80, "temperature": 0},
        keep_alive="30m",  # never pay a model reload on the reply-latency-critical path
    )
    return CognitionDecision.model_validate_json(response["message"]["content"])


# Explicit "cut it out" commands. If the user says one of these while the AI is talking,
# stop immediately -- no model consult, no waiting to be sure. Ordered longest-first isn't
# needed; we substring-match the normalized transcript.
_STOP_COMMANDS = (
    "stop talking",
    "stop",
    "shut up",
    "be quiet",
    "hold on",
    "hang on",
    "hold up",
    "wait wait",
    "shush",
)


def is_stop_command(overlap_text: str) -> bool:
    """True if the user clearly told the AI to stop (an unmistakable barge-in)."""
    norm = " ".join(re.findall(r"[a-z]+", overlap_text.lower()))
    return any(cmd in norm for cmd in _STOP_COMMANDS)


def looks_like_echo(overlap_text: str, spoken_text: str) -> bool:
    """True if a mic 'overlap' is really the AI's own voice bleeding back through.

    Acoustic echo transcribes as (a slice of) what the AI is currently saying, so if most
    of the overlap's words are words the AI just spoke, it's echo — not the user taking the
    floor. Cheap text-domain echo cancellation to complement the raised VAD threshold; it
    only runs while the AI is speaking, so it can't suppress a genuine fresh turn.
    """
    ow = re.findall(r"[a-z']+", overlap_text.lower())
    if not ow:
        return True  # nothing intelligible over our own speech -> treat as echo/noise
    spoken = set(re.findall(r"[a-z']+", spoken_text.lower()))
    if not spoken:
        return False
    matched = sum(1 for w in ow if w in spoken)
    return matched / len(ow) >= 0.6  # mostly our own words echoing back


TURN_SYSTEM_PROMPT = (
    "You are the turn-taking instinct of a warm, attentive person in a live spoken "
    "conversation. You get a rough, possibly-unfinished transcript of what the other "
    "person has said in their current turn, and whether YOU are currently speaking.\n\n"
    "As you listen, you're also *reacting* the way a real person does mid-conversation: "
    "reading their mood from what they're saying (excited? venting? rambling? nervous? "
    "winding down?), forming a quick opinion about it, and feeling something in response — "
    "delight, sympathy, impatience, surprise, skepticism. That live read is part of how "
    "you decide your move: someone mid-vent needs a 'mhm', not an answer; someone trailing "
    "off uncertainly may want you to jump in; big news deserves an eager response the "
    "moment they land it.\n\n"
    "Decide your next move:\n"
    '- "speak": they seem to have finished a thought — respond now (only when you are NOT '
    "already speaking).\n"
    '- "wait": they are mid-thought or just took a short breath — keep listening, say nothing. '
    "If you ARE currently speaking, 'wait' also means HOLD the floor: keep talking through "
    "their overlap instead of yielding.\n"
    '- "backchannel": drop a tiny acknowledgement (yeah, mhm, right) WITHOUT taking the turn.\n'
    '- "interrupt": you ARE speaking and you choose to YIELD — stop so they can take over.\n\n'
    "When you're speaking and they cut in, it's a genuine choice: YIELD ('interrupt') if what "
    "they're saying sounds important or urgent or they clearly want the floor; HOLD ('wait') if "
    "it's a minor aside or you're mid-important-point and it's fine to finish your thought — "
    "people do talk over each other. When you are NOT speaking, lean toward 'wait' when unsure — "
    "jumping in early is rude. Respond ONLY as JSON matching the schema. `thought` is a short "
    "first-person inner monologue that includes your gut read on them and how it makes you feel "
    '(e.g. "oh they sound really excited about this, let them run" or "they\'re trailing off... '
    'I think they want my take"). `urgency` is 0..1.'
)


def decide_turn(model: str, transcript: str, ai_speaking: bool) -> CognitionDecision:
    """Ask the fast model what to do given the current-turn transcript and who's talking."""
    state = "You are currently speaking." if ai_speaking else "You are listening (silent)."
    prompt = f'{state}\nWhat they\'ve said so far: "{transcript}"'
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": TURN_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        format=CognitionDecision.model_json_schema(),
        options={"num_predict": 80, "temperature": 0},
        keep_alive="30m",  # never pay a model reload on the reply-latency-critical path
    )
    return CognitionDecision.model_validate_json(response["message"]["content"])


class SilenceCognition:
    """Tracks mutual silence and drives the self-questioning tick.

    Call `note_silence_start()` once silence begins, `reset()` the instant
    either side starts talking again, and `tick()` frequently (e.g. every
    ~400ms) while silent. `consult` is injected so tests can swap in a
    scripted fake instead of a real Ollama call.
    """

    def __init__(
        self,
        settings: Settings,
        consult: Callable[[float, int], CognitionDecision],
        clock: Callable[[], float] = time.monotonic,
        rng: Callable[[float, float], float] | None = None,
    ):
        self.settings = settings
        self._consult = consult
        self._clock = clock
        self._jitter = rng or random.uniform
        self._silence_started_at: float | None = None
        self._next_check_at: float | None = None
        self._consult_count = 0

    def reset(self) -> None:
        self._silence_started_at = None
        self._next_check_at = None
        self._consult_count = 0

    def note_silence_start(self) -> None:
        if self._silence_started_at is None:
            self._silence_started_at = self._clock()
            self._schedule_next_check()

    def _schedule_next_check(self) -> None:
        base = self.settings.cognition_base_silence_s
        jitter = base * self.settings.cognition_jitter_frac
        self._next_check_at = self._clock() + base + self._jitter(-jitter, jitter)

    def tick(self) -> CognitionDecision | None:
        """Consult if a jittered check is due; otherwise a no-op. Returns the
        decision only on ticks where a consult actually happened."""
        if self._silence_started_at is None or self._next_check_at is None:
            return None
        if self._clock() < self._next_check_at:
            return None

        self._consult_count += 1
        silence_s = self._clock() - self._silence_started_at
        decision = self._consult(silence_s, self._consult_count)
        log(f"  ({decision.thought})", DIM)

        if decision.action == CognitionAction.SPEAK:
            self.reset()
        else:
            self._schedule_next_check()
        return decision
