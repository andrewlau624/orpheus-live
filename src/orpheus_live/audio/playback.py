"""TTS output: a persistent output stream (AudioSink) and sentence-by-sentence speech.

Replies are synthesized incrementally (see `engines.tts.OrpheusVoice.stream`) and pushed
chunk-by-chunk into a single long-lived `sd.OutputStream`. Playing through one persistent
stream — instead of a fresh `sd.play()` per sentence — removes the inter-sentence gap and
lets a barge-in stop the audio *mid-sentence* instantly (`AudioSink.clear()`), while an
underrun simply emits silence (never a repeated beat).
"""

import queue
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator

import numpy as np
import sounddevice as sd

from ..config import Settings
from ..debug import tracer

# Split after . ! ? … (a trailing <tag> stays attached to its sentence).
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])(?:\s*(<[a-z_]+>))?\s+")
_TAG_ONLY = re.compile(r"<[a-z_]+>")

# Orpheus repeats/rambles on very short prompts (a bare "Hmm." often never
# emits end-of-speech), so tiny fragments are merged with their neighbor
# instead of synthesized alone.
MIN_SENTENCE_CHARS = 12


def _spoken_len(sentence: str) -> int:
    return len(_TAG_ONLY.sub("", sentence).strip())


# Orpheus emits ~137.5 tokens/sec of audio and a spoken word is ~55 tokens (see
# OrpheusVoice._max_tokens), so a word is ~0.4s; the pad absorbs short-word bias.
_WORD_SECS = 0.4
_EST_PAD_SECS = 0.5


def estimate_speech_secs(sentence: str) -> float:
    """Rough expected audio duration for a sentence, for lead-aware playback pacing."""
    words = len(_TAG_ONLY.sub(" ", sentence).split())
    return _WORD_SECS * words + _EST_PAD_SECS


class PreSynthStream:
    """A growing buffer of audio chunks produced speculatively while the user talks.

    A background synth task appends chunks (`add`) and marks `finish()`; the consumer
    (`iter_chunks`) yields buffered chunks immediately and blocks for more until the
    stream finishes or is cancelled. Because synthesis starts during the user's speech,
    the head is already buffered when they stop -> first audio is instant, and the
    head-start offsets sub-realtime generation so playback stays ahead (smooth).
    """

    def __init__(self, text: str):
        self.text = text
        self._chunks: list[np.ndarray] = []
        self._done = False
        self._cancelled = False
        self._cv = threading.Condition()

    def add(self, chunk: np.ndarray) -> None:
        with self._cv:
            self._chunks.append(chunk)
            self._cv.notify_all()

    def finish(self) -> None:
        with self._cv:
            self._done = True
            self._cv.notify_all()

    def cancel(self) -> None:
        with self._cv:
            self._cancelled = True
            self._done = True
            self._cv.notify_all()

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def iter_chunks(self) -> Iterator[np.ndarray]:
        i = 0
        while True:
            with self._cv:
                while i >= len(self._chunks) and not self._done:
                    self._cv.wait()
                if i >= len(self._chunks):
                    return  # finished and fully drained
                chunk = self._chunks[i]
                i += 1
            yield chunk


def split_sentences(text: str) -> list[str]:
    """Split a reply into speakable sentences, keeping emotion <tags> attached.

    Fragments shorter than MIN_SENTENCE_CHARS (ignoring tags) are merged into
    the following sentence (or the previous one, at the end) -- short prompts
    make Orpheus glitch and repeat itself.
    """
    parts: list[str] = []
    last = 0
    for m in _SENTENCE_SPLIT.finditer(text):
        chunk = text[last : m.start()] + (f" {m.group(1)}" if m.group(1) else "")
        if chunk.strip():
            parts.append(chunk.strip())
        last = m.end()
    tail = text[last:].strip()
    if tail:
        parts.append(tail)

    merged: list[str] = []
    for part in parts:
        if merged and _spoken_len(merged[-1]) < MIN_SENTENCE_CHARS:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    if len(merged) > 1 and _spoken_len(merged[-1]) < MIN_SENTENCE_CHARS:
        tail_part = merged.pop()
        merged[-1] = f"{merged[-1]} {tail_part}"
    return merged


def iter_stream_sentences(chunks: Iterator[str]) -> Iterator[str]:
    """Yield complete speakable sentences from a streaming text source, as they form.

    The source is drained eagerly on a background thread, so the LLM keeps generating
    (thinking) while already-yielded sentences are being synthesized and spoken. A
    sentence is yielded the moment it ends with terminal punctuation — don't wait for
    the next sentence to start. Short fragments stay merged as `split_sentences` does.
    """
    q: queue.Queue[str | None] = queue.Queue()

    def _drain() -> None:
        try:
            for chunk in chunks:
                q.put(chunk)
        finally:
            q.put(None)  # always unblock the consumer, even if the stream raises

    threading.Thread(target=_drain, daemon=True).start()
    buf = ""
    while True:
        chunk = q.get()
        if chunk is None:
            break
        buf += chunk
        parts = split_sentences(buf)
        # Yield every complete sentence immediately; keep only the still-growing tail.
        if len(parts) > 1:
            yield from parts[:-1]
            buf = parts[-1]
        elif (
            len(parts) == 1
            and buf.rstrip().endswith((".", "!", "?", "…"))
            and _spoken_len(parts[0]) >= MIN_SENTENCE_CHARS
        ):
            # One complete, long-enough sentence -> speak it now, don't wait for sentence 2.
            # (Short fragments like "Wow." stay buffered so split_sentences can merge them;
            # Orpheus glitches/repeats on tiny standalone prompts.)
            yield parts[0]
            buf = ""
    # Drain whatever's left (the final sentence or fragment).
    yield from split_sentences(buf)


class AudioSink:
    """A persistent mono output stream fed by variable-length float32 chunks.

    Writes are tagged with an epoch (`begin(epoch)` opens one). `clear()` bumps the
    epoch, drops buffered audio, and unblocks any waiting writer — so a barge-in cuts
    playback instantly and in-flight writes for the old epoch become no-ops. The
    callback zero-fills on underrun, so falling briefly behind is a short silence
    rather than the library's repeated-beat glitch.

    Playback start is *lead-aware*: `pace(expected_s)` announces roughly how much audio
    the current sentence will produce, and `write()` measures the source's live delivery
    rate r (audio secs per wall sec). A sentence of T seconds from a rate-r source plays
    through gap-free iff ~(1-r)/r of the *remaining* audio is buffered before starting —
    zero hold when r >= 1 (fast source: start at the small prebuffer), and the minimum
    possible hold when r < 1. The measured rate is remembered session-wide (EMA), so
    later sentences arm on real numbers immediately instead of re-holding their full
    estimate while the rate is re-learned.

    Running dry at a sentence seam (the next sentence is paced, or this one delivered
    ~its estimate) is the natural place to wait and is NOT an underrun. A genuine
    mid-sentence stall disarms, re-applies the rule with fresher numbers, and raises
    the session-sticky floor to `tts_rebuffer_s` — one audible pause per stall instead
    of machine-gun mid-word chop.
    """

    _FADE = 240  # samples of fade-in at the start of an utterance (declick)
    _RATE_MIN_ELAPSED = 0.05  # secs of arrivals before the measured rate is trusted
    _RATE_EMA_ALPHA = 0.5  # blend of the newest segment's rate into the session estimate
    _RATE_PESSIMISM = 0.85  # plan for the source running a bit slower than measured:
    # the rate dips when STT/cognition contend for the machine mid-sentence (observed
    # 0.56 -> 0.38), and an optimistic arm is a mid-word stall.
    _MARGIN_SECS = 0.25  # safety cushion on top of the computed deficit
    _SEG_DONE_FRAC = 0.8  # segment this close to its estimate is "delivered", not stalled
    _EST_SCALE_ALPHA = 0.4  # blend of the newest delivered/estimated ratio (calibration)
    _EST_SCALE_MIN, _EST_SCALE_MAX = 0.5, 1.5  # calibration can't run away on outliers
    _EST_MIN_DELIVERED_S = 0.5  # don't calibrate on tiny segments

    def __init__(self, settings: Settings):
        self._sr = settings.tts_sample_rate
        self._prebuffer = int(settings.tts_prebuffer_s * self._sr)
        self._rebuffer = int(settings.tts_rebuffer_s * self._sr)
        self._max_hold = int(settings.tts_max_hold_s * self._sr)  # cap on the arm lead
        self._fade_len = max(1, int(settings.tts_interrupt_fade_ms / 1000 * self._sr))
        self._fade_left = 0  # samples remaining in an in-progress interrupt fade-out
        self._base_cap = int(2.0 * self._sr)  # steady-state backpressure cap (~2s ahead)
        self._max_buffered = self._base_cap  # raised by pace() so a held sentence fits
        self._buf: deque[np.ndarray] = deque()
        self._head = 0  # read cursor into buf[0]
        self._buffered = 0  # total samples across buf
        self._epoch = 0
        self._armed = False  # gate playback until the arm target is met
        self._draining = False  # flush() in progress: an empty buffer is the end, not a stall
        self._faded = False  # fade-in applied to this utterance yet?
        self._had_underrun = False  # session-sticky: floor the arm target at _rebuffer
        self._on_audible: Callable[[], None] | None = None  # fired once, when audio starts
        self._expected = 0  # samples the current paced sentence is expected to produce
        self._seg_t0: float | None = None  # first-write time of the paced segment
        self._seg_t_last: float | None = None  # last-write time (rate spans arrivals only)
        self._seg_written = 0  # samples written since pace() (rate numerator)
        self._seg_base = 0  # samples in the first write (excluded from the rate estimate)
        self._rate_ema: float | None = None  # session-wide source rate: sentences arrive
        # from the same engine/link, so N+1 arms on N's measured speed instead of
        # re-holding its full estimate while the rate is re-learned from scratch.
        self._est_scale = 1.0  # session calibration of the text-length duration estimate:
        # word-count estimates ran ~1.4x over measured audio, and hold = (1-r)*estimate,
        # so systematic overshoot directly inflates every pause. Learned from completed
        # segments (delivered vs estimated), applied to future pace() calls.
        self._seg_raw_expected = 0  # this segment's UNscaled estimate (calibration basis)
        self._boundary = False  # pace() ran for the next segment: a dry-out is a seam
        self._disarmed_at: float | None = None  # when playback last stopped (gap timing)
        self._disarm_cause = "start"  # start | seam | UNDERRUN — why we're not playing
        self._cv = threading.Condition()
        # Larger blocksize + "high" device latency give the callback a looser deadline, so
        # it keeps feeding audio through GIL stalls / CPU thrash instead of crackling.
        self._stream = sd.OutputStream(
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            blocksize=settings.tts_output_blocksize,
            latency="high",
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, outdata, frames, time_info, status) -> None:
        out = outdata.reshape(-1)
        with self._cv:
            if self._fade_left > 0:
                # Interrupt fade-out: keep pulling real audio but ramp it to zero over
                # ~tts_interrupt_fade_ms, so a barge-in sounds like the voice halting/
                # trailing off rather than a hard click. When the ramp completes, go silent.
                filled = 0
                while filled < frames and self._buf:
                    head = self._buf[0]
                    take = min(frames - filled, head.shape[0] - self._head)
                    out[filled : filled + take] = head[self._head : self._head + take]
                    filled += take
                    self._head += take
                    self._buffered -= take
                    if self._head >= head.shape[0]:
                        self._buf.popleft()
                        self._head = 0
                out[filled:] = 0.0
                g0 = self._fade_left / self._fade_len
                ramp = np.maximum(
                    0.0, g0 - np.arange(1, frames + 1, dtype=np.float32) / self._fade_len
                )
                out *= ramp
                self._fade_left = max(0, self._fade_left - frames)
                if self._fade_left == 0:
                    self._buf.clear()
                    self._head = self._buffered = 0
                    self._armed = False
                self._cv.notify_all()
                return
            if not self._armed:
                out[:] = 0.0
                return
            filled = 0
            while filled < frames and self._buf:
                head = self._buf[0]
                take = min(frames - filled, head.shape[0] - self._head)
                out[filled : filled + take] = head[self._head : self._head + take]
                filled += take
                self._head += take
                self._buffered -= take
                if self._head >= head.shape[0]:
                    self._buf.popleft()
                    self._head = 0
            if filled < frames:
                out[filled:] = 0.0  # underrun -> silence, never a repeat
                if not self._draining:
                    # Ran dry: stop and re-buffer (write() re-arms once the lead-aware
                    # target is met again) instead of dribbling out each chunk as it
                    # lands -> one pause, not stuttered words. Only a genuine
                    # mid-segment stall raises the sticky floor; running out at a
                    # sentence seam (next segment already paced, or this one delivered
                    # ~its estimate) is the natural place to wait.
                    self._armed = False
                    seam = self._boundary or (
                        self._expected > 0
                        and self._seg_written >= self._SEG_DONE_FRAC * self._expected
                    )
                    if not seam:
                        self._had_underrun = True
                    self._disarmed_at = time.monotonic()
                    self._disarm_cause = "seam" if seam else "UNDERRUN"
                    tracer.emit(
                        "sink.dry",
                        cause=self._disarm_cause,
                        delivered_s=self._seg_written / self._sr,
                        expected_s=self._expected / self._sr,
                    )
            self._cv.notify_all()

    def begin(self, epoch: int, on_audible: Callable[[], None] | None = None) -> None:
        """Open a new utterance: adopt `epoch`, drop leftover audio, reset pacing state.

        `on_audible` fires exactly once, the moment playback actually starts (arming) —
        not when the first chunk is merely buffered — so mic-unmute tracks audibility.
        The session-wide rate estimate survives: the engine/link doesn't change between
        utterances, so the next one arms on the measured speed, not from scratch.
        """
        with self._cv:
            self._fold_segment_rate(completed=False)
            self._epoch = epoch
            self._buf.clear()
            self._head = self._buffered = 0
            self._armed = False
            self._faded = False
            self._fade_left = 0  # cancel any interrupt fade still draining
            self._on_audible = on_audible
            self._expected = 0
            self._seg_raw_expected = 0
            self._seg_t0 = self._seg_t_last = None
            self._seg_written = 0
            self._seg_base = 0
            self._boundary = False
            self._disarmed_at = time.monotonic()
            self._disarm_cause = "start"
            self._max_buffered = self._base_cap
            self._cv.notify_all()

    def pace(self, expected_s: float) -> None:
        """Announce the next SENTENCE's expected audio duration (call before its writes).

        Each sentence is its own paced segment: it gets a lead sized to play *that*
        sentence through without underrunning mid-word. The buffer carries across the
        boundary, so on a source fast enough to stay ahead there's no re-hold (playback
        is continuous); only when the buffer has actually drained by the boundary does the
        next sentence re-arm — a pause at a natural sentence break, never mid-word chop.
        Folds the finished sentence's rate into the session estimate and widens the
        backpressure cap so a full sentence can be buffered while it's held.
        """
        with self._cv:
            self._fold_segment_rate(completed=True)
            raw = int(max(0.0, expected_s) * self._sr)
            self._seg_raw_expected = raw
            self._expected = int(raw * self._est_scale)
            self._seg_t0 = self._seg_t_last = None
            self._seg_written = 0
            self._seg_base = 0
            self._boundary = True
            self._max_buffered = max(self._base_cap, self._expected + self._sr)
            tracer.emit(
                "sink.pace",
                est_s=expected_s,
                est_scale=self._est_scale,
                rate_ema=self._rate_ema,
            )
            self._cv.notify_all()

    def _fold_segment_rate(self, completed: bool = False) -> None:
        """Fold the closing segment into the session estimates (rate and calibration).

        The rate spans first arrival to last arrival — idle time after a segment's
        final chunk (e.g. waiting on the LLM for the next sentence) is not the
        source being slow, so it must not dilute the estimate.

        Only `completed` segments (closed by pace()/flush(), not a barge-in clear())
        calibrate the duration estimator: a cancelled sentence delivered less audio
        than estimated because it was cut short, not because the estimate was high.
        """
        if self._seg_t0 is None or self._seg_t_last is None:
            return
        r = None
        elapsed = self._seg_t_last - self._seg_t0
        measured = self._seg_written - self._seg_base
        if elapsed >= self._RATE_MIN_ELAPSED and measured > 0:
            r = (measured / self._sr) / elapsed
            a = self._RATE_EMA_ALPHA
            self._rate_ema = r if self._rate_ema is None else a * r + (1 - a) * self._rate_ema
        if (
            completed
            and self._seg_raw_expected > 0
            and self._seg_written >= self._EST_MIN_DELIVERED_S * self._sr
        ):
            ratio = self._seg_written / self._seg_raw_expected
            a = self._EST_SCALE_ALPHA
            scale = a * ratio + (1 - a) * self._est_scale
            self._est_scale = min(self._EST_SCALE_MAX, max(self._EST_SCALE_MIN, scale))
        tracer.emit(
            "sink.segment",
            delivered_s=self._seg_written / self._sr,
            expected_s=self._expected / self._sr,
            rate=r,
            rate_ema=self._rate_ema,
            est_scale=self._est_scale,
        )

    def _arm_target(self, now: float) -> int:
        """Samples that must be buffered before (re)starting playback, gap-free.

        To play E seconds of audio at 1x while it arrives at r < 1, the buffered lead
        must cover the shortfall over the whole playout: the exact minimum is (1-r)*E
        (arrive-rate r for E/r wall-secs vs. play-rate 1 for E secs). We compare that
        against total buffered, plus a small margin for jitter/discretization. r comes
        from this segment's arrivals once there are enough, else the session estimate
        (warmup + earlier sentences); with neither, hold the full estimate.
        """
        floor = self._rebuffer if self._had_underrun else self._prebuffer
        if not self._expected:
            return floor
        elapsed = 0.0 if self._seg_t0 is None else (now - self._seg_t0)
        # Rate is measured from audio delivered *after* the first write: that write's
        # samples arrived at t0 with zero elapsed time, so counting them inflates the
        # rate and arms too early.
        measured = self._seg_written - self._seg_base
        if elapsed >= self._RATE_MIN_ELAPSED and measured > 0:
            r = (measured / self._sr) / elapsed
        elif self._rate_ema is not None:
            r = self._rate_ema
        else:
            return max(floor, self._expected)
        # Plan for the source running a bit slower than measured — a dip mid-sentence
        # (STT/cognition contention, network jitter) would otherwise stall mid-word.
        r *= self._RATE_PESSIMISM
        if r >= 1.0:
            return floor
        lead = int((1.0 - r) * self._expected)
        if lead <= 0:
            return floor
        target = lead + int(self._MARGIN_SECS * self._sr)
        # Cap the wait: on a slow source the ideal lead for a long sentence is several
        # seconds of pre-buffering (dead air between sentences). Prefer to start talking
        # within _max_hold and let the rebuffer turn any resulting mid-sentence dip into
        # one clean pause -- a conversation reads far better fast-with-a-hiccup than
        # smooth-after-silence. On a fast source `lead` is ~0 and this never binds.
        return max(floor, min(target, self._max_hold))

    def _fire_audible(self) -> None:
        cb, self._on_audible = self._on_audible, None
        if cb is not None:
            cb()

    def write(self, chunk: np.ndarray, epoch: int) -> None:
        """Append a chunk (blocking while the buffer is full); a no-op if `epoch` is stale."""
        chunk = np.clip(np.asarray(chunk, dtype=np.float32).reshape(-1), -1.0, 1.0)
        if chunk.size == 0:
            return
        with self._cv:
            if epoch != self._epoch:
                return
            if not self._faded:
                f = min(self._FADE, chunk.shape[0])
                chunk = chunk.copy()
                chunk[:f] *= np.linspace(0.0, 1.0, f, dtype=np.float32)
                self._faded = True
            while self._buffered > self._max_buffered and epoch == self._epoch:
                self._cv.wait()
            if epoch != self._epoch:
                return
            now = time.monotonic()
            if self._seg_t0 is None:
                self._seg_t0 = now
                self._seg_base = chunk.shape[0]  # first burst: excluded from rate
            self._seg_t_last = now
            self._seg_written += chunk.shape[0]
            self._buf.append(chunk)
            self._buffered += chunk.shape[0]
            tracer.emit(
                "sink.write",
                _echo=False,
                samples=chunk.shape[0],
                buffered_s=self._buffered / self._sr,
                armed=self._armed,
            )
            if not self._armed and self._buffered >= self._arm_target(now):
                self._armed = True
                self._boundary = False  # playing again: the next dry-out is a real stall
                tracer.emit(
                    "sink.arm",
                    cause=self._disarm_cause,
                    gap_s=None if self._disarmed_at is None else now - self._disarmed_at,
                    hold_s=None if self._seg_t0 is None else now - self._seg_t0,
                    buffered_s=self._buffered / self._sr,
                    target_s=self._arm_target(now) / self._sr,
                )
                self._fire_audible()
            self._cv.notify_all()

    def flush(self, epoch: int) -> None:
        """Block until buffered audio for `epoch` has fully played (or the epoch is cancelled)."""
        with self._cv:
            # Short utterances may never reach the arm target; and while draining, an
            # empty buffer means "done", so the callback must not treat it as a stall.
            tracer.emit("sink.flush", buffered_s=self._buffered / self._sr)
            self._fold_segment_rate(completed=True)  # bank the last sentence's numbers
            self._seg_t0 = self._seg_t_last = None
            self._seg_raw_expected = 0
            self._armed = True
            self._draining = True
            if self._faded:  # only audible if something was actually written
                self._fire_audible()
            self._cv.notify_all()
            try:
                while self._buffered > 0 and epoch == self._epoch:
                    self._cv.wait()
            finally:
                # Drained (or cancelled): disarm so the callbacks that fire before the
                # next begin() take the silent early-return path, not the underrun branch
                # — an intentionally emptied buffer must not be read as a stall.
                self._draining = False
                if epoch == self._epoch:
                    self._armed = False

    def clear(self) -> None:
        """Stop for barge-in: invalidate the epoch now, fade the in-flight audio to silence.

        The epoch bump is immediate so the cancelled generation's further writes are
        dropped, but the audio already buffered is ramped out over tts_interrupt_fade_ms
        (see `_callback`) instead of cut dead — a natural halt, not a click. `begin()` for
        the next reply cancels any fade still in progress.
        """
        with self._cv:
            tracer.emit("sink.clear", dropped_s=self._buffered / self._sr)
            self._epoch += 1
            self._on_audible = None
            # Fade only if there's actually audible audio mid-flight; otherwise drop clean.
            if self._armed and self._buffered > 0:
                self._fade_left = min(self._fade_len, self._buffered)
            else:
                self._buf.clear()
                self._head = self._buffered = 0
                self._armed = False
                self._fade_left = 0
            # A cancelled segment delivered less than estimated because it was cut
            # short — drop it entirely so it can't skew calibration or the rate EMA.
            self._expected = 0
            self._seg_raw_expected = 0
            self._seg_t0 = self._seg_t_last = None
            self._seg_written = 0
            self._seg_base = 0
            self._cv.notify_all()

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


class SpeechPlayer:
    """Synthesizes and plays a reply, cancellable mid-utterance.

    Two modes (chosen by `buffer_whole`):

    - streaming (default): write each sentence's chunks as they're produced. The sink's
      lead-aware pacing holds back just enough audio that even a sub-realtime source
      plays through smoothly (see AudioSink), so TTFB is as low as the source allows.
    - buffered: synthesize the ENTIRE reply first, then play it. Nothing generates while
      audio plays, so the real-time audio callback never fights MLX for the GIL --
      fallback for local setups where streaming still crackles under load.

    `cancel()` stops both further generation (via the gen-id checked between chunks) and
    playback (via `sink.clear()`), enabling a clean mid-utterance stop in either mode.
    """

    def __init__(
        self,
        stream: Callable[[str], Iterator[np.ndarray]],
        sink: AudioSink,
        buffer_whole: bool = False,
        on_first_audio: Callable[[], None] | None = None,
    ):
        self._stream = stream
        self._sink = sink
        self._buffer_whole = buffer_whole
        # Fired once per speak(), when the sink actually starts playing (arming) -- used by
        # the orchestrator to un-mute the mic (lag-aware pickup) exactly at audibility.
        self._on_first_audio = on_first_audio
        self._gen_id = 0
        # Serializes all Orpheus generation (playback + speculative pre-synth share MLX).
        self.synth_lock = threading.Lock()

    def cancel(self) -> None:
        """Abort playback now and discard any not-yet-played audio."""
        self._gen_id += 1
        self._sink.clear()

    def _iter_sentence(self, sentence: str, my_gen: int):
        """Yield a sentence's audio chunks (honoring cancellation), closing the generator."""
        chunks = self._stream(sentence)
        try:
            for chunk in chunks:
                if self._gen_id != my_gen:
                    return
                yield chunk
        finally:
            close = getattr(chunks, "close", None)
            if close is not None:
                close()

    def speak(self, text: str) -> None:
        """Synthesize and play `text`; bails early if cancel() runs concurrently."""
        my_gen = self._gen_id
        self._sink.begin(my_gen, self._on_first_audio)
        sentences = split_sentences(text)
        if self._buffer_whole:
            # Synthesize the whole reply BEFORE playing any of it, so no generation runs
            # while audio plays (the callback never fights MLX for the GIL -> smooth).
            clips: list[np.ndarray] = []
            for sentence in sentences:
                if self._gen_id != my_gen:
                    break
                with self.synth_lock:
                    clips.extend(self._iter_sentence(sentence, my_gen))
            for clip in clips:
                if self._gen_id != my_gen:
                    break
                self._sink.write(clip, my_gen)
        else:
            # Streaming: pace EACH sentence so it gets a lead sized to play through
            # without underrunning mid-word. The buffer carries across boundaries, so a
            # fast-enough source never re-holds (continuous); a slow one pauses at the
            # sentence break, not inside a word.
            for sentence in sentences:
                if self._gen_id != my_gen:
                    break
                self._sink.pace(estimate_speech_secs(sentence))
                tracer.emit("synth.sentence_start", _echo=False, text=sentence)
                t0 = time.monotonic()
                with self.synth_lock:
                    for chunk in self._iter_sentence(sentence, my_gen):
                        self._sink.write(chunk, my_gen)
                tracer.emit("synth.sentence_done", wall_s=time.monotonic() - t0)
        self._sink.flush(my_gen)

    def speak_stream(self, text_stream: Iterator[str]) -> str:
        """Speak a reply while it is still being generated (thinking ∥ speaking).

        Sentences are consumed from `text_stream` as they complete, so the first
        sentence's audio starts long before the LLM finishes the reply. Each sentence is
        paced so it gets a lead sized to play through without underrunning mid-word; the
        buffer carries across boundaries, so a fast-enough source plays continuously and a
        slow one pauses at the sentence break rather than chopping inside a word.

        Returns the full reply text (everything actually generated), even if playback
        was cancelled partway, so the caller can still log/remember it.
        """
        my_gen = self._gen_id
        self._sink.begin(my_gen, self._on_first_audio)
        spoken: list[str] = []
        for sentence in iter_stream_sentences(text_stream):
            spoken.append(sentence)  # collect even when cancelled: memory wants the text
            if self._gen_id != my_gen:
                continue  # cancelled: keep draining the LLM cheaply, synthesize nothing
            if not self._buffer_whole:
                self._sink.pace(estimate_speech_secs(sentence))
            tracer.emit("synth.sentence_start", _echo=False, text=sentence)
            t0 = time.monotonic()
            with self.synth_lock:
                for chunk in self._iter_sentence(sentence, my_gen):
                    self._sink.write(chunk, my_gen)
            tracer.emit("synth.sentence_done", wall_s=time.monotonic() - t0)
        self._sink.flush(my_gen)
        return " ".join(spoken)
