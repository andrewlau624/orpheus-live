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
from ..models import CognitionAction, CognitionDecision, OverlapVerdict

COGNITION_SYSTEM_PROMPT = (
    "You are the inner monologue of one participant in a live spoken "
    "conversation, during a lull where nobody is talking. You're a real "
    "person sitting with someone in silence -- and silence FEELS like "
    "something: comfortable, awkward, heavy, companionable. Decide whether "
    "to break it.\n\n"
    "You'll be told how long the silence has lasted and how many times "
    "you've already checked in on it. The longer it drags on and the more "
    "checks that have already passed, the more natural it is to lean toward "
    "speaking -- like a person's own hesitation fading. Don't always wait, "
    "and don't jump in instantly either -- vary it.\n\n"
    "Let your thought carry a real feeling about the moment: restlessness "
    '("ugh, this is getting awkward"), curiosity ("I wonder what they\'re '
    'thinking about"), contentment ("honestly this quiet is kind of nice"), '
    'or mild worry ("did I say something weird earlier?"). React like a '
    "person, not a timer.\n\n"
    "Respond ONLY as JSON matching the schema. `thought` is a short, "
    "first-person inner monologue with genuine emotion in it. "
    '`action` is "speak" or "wait" ("backchannel"/"interrupt" aren\'t '
    "available yet). `urgency` is 0..1, how strongly you feel about it."
)


def consult(model: str, silence_s: float, consult_count: int) -> CognitionDecision:
    """Ask the dedicated cognition model for a structured silence decision."""
    prompt = f"Silence has lasted {silence_s:.1f}s. This is check-in #{consult_count}."
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": COGNITION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        format=CognitionDecision.model_json_schema(),
        options={"num_predict": 80},
    )
    return CognitionDecision.model_validate_json(response["message"]["content"])


# Words that, alone or in tiny combinations, are almost always a backchannel
# ("yeah", "oh wow", "mm right") rather than an attempt to take the turn. Includes
# fillers and common Whisper hallucinations on noise/short sounds, so agreeing
# murmurs and background blips don't get mistaken for a turn-grab.
_BACKCHANNEL_WORDS = frozenset(
    "yeah yes yep yup ya yea right true okay ok k sure mm mhm mmhm mmhmm hmm hm "
    "uh uhhuh um umm erm huh wow oh ohh ooh ah aw aha haha ha lol nice cool "
    "totally exactly really interesting gotcha damn whoa woah nah welp ope so "
    "no way".split()
)


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


def quick_overlap_verdict(overlap_text: str) -> OverlapVerdict | None:
    """Instant heuristic tier of the overlap classifier (no model call).

    Returns a verdict when the transcript is unambiguous, or None for the gray
    zone where the cognition model should decide.
    """
    words = re.findall(r"[a-z']+", overlap_text.lower())
    if not words:
        return OverlapVerdict.BACKCHANNEL  # nothing intelligible -> don't stop for it
    if is_stop_command(overlap_text):
        return OverlapVerdict.INTERRUPT  # explicit "stop" -> cut in now
    if len(words) >= 8:
        return OverlapVerdict.INTERRUPT  # nobody backchannels a whole sentence
    if len(words) <= 4 and all(w.strip("'") in _BACKCHANNEL_WORDS for w in words):
        return OverlapVerdict.BACKCHANNEL
    return None


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
        options={"num_predict": 80},
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
