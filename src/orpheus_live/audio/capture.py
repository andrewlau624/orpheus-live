"""Microphone capture and continuous, VAD-gated speech segmentation.

The mic is captured 24/7. VAD is only a cheap *speech-presence* gate now -- it no longer
decides turn-taking. It accumulates the current turn's audio (with a little pre-roll) and,
whenever the talker takes a beat of silence, fires `on_pause` so the orchestrator can ask
the cognition model what to do (respond / keep waiting / interrupt / backchannel). A longer
silence fires `on_pause(final=True)` as a safety net so a turn can never hang unanswered.

`_process_frame` takes one frame at a time and is called directly (no thread, no queue) so
tests can drive it deterministically.
"""

import queue
import threading
import time
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from ..config import Settings
from ..console import DIM, log
from ..debug import tracer
from .vad import Vad


def _frames_to_audio(frames: list[bytes]) -> np.ndarray | None:
    """int16 PCM frames -> float32 waveform (or None if empty)."""
    if not frames:
        return None
    pcm = b"".join(frames)
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


class MuteGate:
    """Lag-aware mic mute with a watchdog: force-unmutes after `max_mute_s`.

    The mute covers the commit-to-reply -> first-audible gap so think-gap noise can't
    spawn cognition. That gap was designed to be ~1-2s; on slow synthesis it can reach
    10s+, during which the user can't barge in or say "stop". The watchdog caps how
    long the mic can be deaf: if audio still isn't audible after `max_mute_s`, unmute
    anyway and accept the noise risk. A generation counter makes stale timers no-ops.
    Duck-types threading.Event's set/clear/is_set, so AudioIn takes either.
    """

    def __init__(self, max_mute_s: float = 0.0):
        self._event = threading.Event()
        self._max = max_mute_s
        self._gen = 0
        self._lock = threading.Lock()

    def set(self) -> None:
        with self._lock:
            self._gen += 1
            gen = self._gen
            self._event.set()
        if self._max > 0:
            t = threading.Timer(self._max, self._timeout, args=(gen,))
            t.daemon = True
            t.start()

    def _timeout(self, gen: int) -> None:
        with self._lock:
            if gen != self._gen or not self._event.is_set():
                return  # already unmuted, or a newer mute owns the mic now
            self._event.clear()
        tracer.emit("mic.unmuted", cause="watchdog", max_mute_s=self._max)

    def clear(self) -> None:
        with self._lock:
            self._gen += 1  # invalidate any pending watchdog
            self._event.clear()

    def is_set(self) -> bool:
        return self._event.is_set()


class AudioIn:
    """Mic capture -> continuous turn accumulation with pause-triggered decision hooks.

    Frame processing runs on its own persistent thread (started by `start()`), so the mic
    keeps draining whether or not the AI is speaking -- interruptions are captured, not
    dropped. `on_pause(turn_audio, final)` fires when the talker pauses; the orchestrator
    decides what the pause means. `reset_turn()` clears the buffer once a turn is consumed.
    """

    def __init__(
        self,
        settings: Settings,
        vad: Vad,
        speaking: threading.Event,
        speak_done_at: list[float],
        user_speaking: threading.Event | None = None,
        on_pause: Callable[[np.ndarray, bool], None] | None = None,
        muted: "threading.Event | MuteGate | None" = None,
        clock: Callable[[], float] = time.time,
    ):
        self.settings = settings
        self.vad = vad
        self.speaking = speaking
        self.speak_done_at = speak_done_at
        self.user_speaking = user_speaking or threading.Event()
        self.on_pause = on_pause
        # When set, the mic is fully ignored (lag-aware pickup): the AI has committed to
        # speak but no audio is out yet, so pre-speech noise must not spawn cognition.
        self.muted = muted or threading.Event()
        self.clock = clock
        self._frame_q: queue.Queue[bytes] = queue.Queue()
        self._stream: sd.InputStream | None = None

        self._triggered = False
        self._voiced_run = 0
        self._silence_run = 0
        self._ring: list[bytes] = []  # recent frames while not triggered (pre-roll)
        self._turn: list[bytes] = []  # audio of the current turn so far
        self._pause_fired = False  # short-pause event already fired for this silence
        self._end_fired = False  # long-silence (final) event already fired for this turn

    def _callback(self, indata, frames, time_info, status) -> None:
        self._frame_q.put(bytes(indata))  # indata: int16, shape (frames, 1)

    def start(self) -> None:
        s = self.settings
        self._stream = sd.InputStream(
            samplerate=s.mic_sample_rate,
            channels=1,
            dtype="int16",
            blocksize=s.frame_len,
            callback=self._callback,
        )
        self._stream.start()
        threading.Thread(target=self._process_loop, daemon=True).start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def turn_audio(self) -> np.ndarray | None:
        """A copy of the current turn's audio so far, or None if no turn is active.

        The `list(...)` snapshot is atomic under the GIL, so it's safe to call from a
        different thread than the one appending frames (used for speculation + decisions).
        """
        if not self._triggered:
            return None
        return _frames_to_audio(list(self._turn))

    def reset_turn(self) -> None:
        """Discard the current turn (call once it's been consumed or dismissed)."""
        self._triggered = False
        self.user_speaking.clear()
        self._voiced_run = self._silence_run = 0
        self._ring.clear()
        self._turn.clear()
        self._pause_fired = self._end_fired = False

    def _process_loop(self) -> None:
        """Continuously drain mic frames; runs for the process's lifetime."""
        while True:
            self._process_frame(self._frame_q.get())

    def _fire_pause(self, final: bool) -> None:
        audio = _frames_to_audio(list(self._turn))
        if audio is not None and self.on_pause is not None:
            self.on_pause(audio, final)

    def _process_frame(self, frame: bytes) -> None:
        s = self.settings
        start_frames = max(1, s.start_speech_ms // s.frame_ms)
        pause_frames = max(1, s.turn_pause_ms // s.frame_ms)
        end_frames = max(1, s.turn_end_ms // s.frame_ms)
        min_frames = max(1, s.min_utterance_ms // s.frame_ms)

        # Lag-aware pickup: while muted, drain and discard the mic entirely. Drop any partial
        # turn so pre-speech noise/breath during the "thinking" gap can't spawn cognition.
        # (Return before touching VAD so no frame is consumed while muted.)
        if self.muted.is_set():
            if self._triggered:
                tracer.emit("mic.turn_dropped_muted")
                self.reset_turn()
            self._ring.clear()
            self._voiced_run = 0
            return

        # Echo suppression: while the AI is speaking, raise the VAD threshold so its own voice
        # bleeding through the speakers doesn't false-trigger as user speech (acoustic echo
        # cancellation via adaptive thresholding). Set to 0 to fall back to the base threshold.
        threshold = (
            s.vad_threshold_during_ai_speech
            if self.speaking.is_set() and s.vad_threshold_during_ai_speech
            else s.vad_threshold
        )
        is_speech = self.vad.is_speech(frame, threshold=threshold)

        if not self._triggered:
            self._ring.append(frame)
            if len(self._ring) > start_frames * 2:
                self._ring.pop(0)
            # Ignore the AI's own trailing audio just after it stops (echo guard).
            if (
                not self.speaking.is_set()
                and (self.clock() - self.speak_done_at[0]) < s.post_speak_cooldown
            ):
                self._voiced_run = 0
                return
            self._voiced_run = self._voiced_run + 1 if is_speech else 0
            if self._voiced_run >= start_frames:
                self._triggered = True
                tracer.emit("mic.speech_start")
                tracer.mark("speech_start")
                self.user_speaking.set()
                self._turn = list(self._ring)  # include the pre-roll
                self._ring.clear()
                self._silence_run = 0
                self._pause_fired = self._end_fired = False
                log("  (listening...)", DIM)
            return

        self._turn.append(frame)
        if is_speech:
            self._silence_run = 0
            # Fresh speech ends the previous silence, so re-arm BOTH pause triggers: the
            # next silence is a new pause that must be able to fire its short-pause consult
            # AND its long-silence safety net. (Leaving _end_fired latched here is what made
            # later sub-utterances hang forever when cognition kept saying "wait".)
            self._pause_fired = False
            self._end_fired = False
            self.user_speaking.set()
            return

        self._silence_run += 1
        enough = len(self._turn) >= min_frames
        # A beat of silence with enough audio banked -> let cognition judge the turn.
        if self._silence_run >= pause_frames and not self._pause_fired and enough:
            self._pause_fired = True
            tracer.emit("mic.pause", turn_s=len(self._turn) * s.frame_ms / 1000)
            self._fire_pause(final=False)
        # A long silence -> force a turn end so nothing can hang unanswered.
        if self._silence_run >= end_frames and not self._end_fired:
            self._end_fired = True
            tracer.emit("mic.turn_end", turn_s=len(self._turn) * s.frame_ms / 1000)
            self._fire_pause(final=True)
