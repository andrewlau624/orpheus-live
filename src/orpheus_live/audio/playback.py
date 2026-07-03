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
    possible hold when r < 1. This is what makes streaming playback smooth even on
    sub-realtime sources, where any fixed-size buffer eventually drains dry.

    Underruns (rate mis-estimated, network stall) disarm and re-apply the same rule with
    fresher numbers, floored at `tts_rebuffer_s` — one audible pause per stall instead of
    machine-gun mid-word chop. The raised floor is sticky for the session.
    """

    _FADE = 240  # samples of fade-in at the start of an utterance (declick)
    _RATE_MIN_ELAPSED = 0.05  # secs of arrivals before the measured rate is trusted
    _RATE_FLOOR = 0.05  # avoid absurd targets from a near-zero measured rate
    _MARGIN_SECS = 0.25  # safety cushion on top of the computed deficit

    def __init__(self, settings: Settings):
        self._sr = settings.tts_sample_rate
        self._prebuffer = int(settings.tts_prebuffer_s * self._sr)
        self._rebuffer = int(settings.tts_rebuffer_s * self._sr)
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
        self._seg_written = 0  # samples written since pace() (rate numerator)
        self._seg_base = 0  # samples in the first write (excluded from the rate estimate)
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
                    # Ran dry mid-stream: stop and re-buffer (write() re-arms once the
                    # lead-aware target is met again) instead of dribbling out each
                    # chunk as it lands -> one pause, not stuttered words.
                    self._armed = False
                    self._had_underrun = True
            self._cv.notify_all()

    def begin(self, epoch: int, on_audible: Callable[[], None] | None = None) -> None:
        """Open a new utterance: adopt `epoch`, drop leftover audio, reset pacing state.

        `on_audible` fires exactly once, the moment playback actually starts (arming) —
        not when the first chunk is merely buffered — so mic-unmute tracks audibility.
        """
        with self._cv:
            self._epoch = epoch
            self._buf.clear()
            self._head = self._buffered = 0
            self._armed = False
            self._faded = False
            self._on_audible = on_audible
            self._expected = 0
            self._seg_t0 = None
            self._seg_written = 0
            self._seg_base = 0
            self._max_buffered = self._base_cap
            self._cv.notify_all()

    def pace(self, expected_s: float) -> None:
        """Announce the next segment's expected audio duration (call before its writes).

        Resets the segment rate measurement and widens the backpressure cap so a
        sentence held back for lead can be buffered in full if need be.
        """
        with self._cv:
            self._expected = int(max(0.0, expected_s) * self._sr)
            self._seg_t0 = None
            self._seg_written = 0
            self._seg_base = 0
            self._max_buffered = max(self._base_cap, self._expected + self._sr)
            self._cv.notify_all()

    def _arm_target(self, now: float) -> int:
        """Samples that must be buffered before (re)starting playback, gap-free.

        With measured source rate r and E samples of the paced segment still to come,
        playback drains the buffer at (1-r) while the source refills at r, so starting
        needs a lead of (1-r)/r * E. No pace() estimate or no rate yet -> fall back to
        the plain floor / the full estimate (the next write lands in ~a chunk anyway).
        """
        floor = self._rebuffer if self._had_underrun else self._prebuffer
        if not self._expected:
            return floor
        elapsed = 0.0 if self._seg_t0 is None else (now - self._seg_t0)
        # Rate is measured from audio delivered *after* the first write: that write's
        # samples arrived at t0 with zero elapsed time, so counting them inflates the
        # rate and arms too early. Until a second burst lands, hold the full estimate.
        measured = self._seg_written - self._seg_base
        if elapsed < self._RATE_MIN_ELAPSED or measured <= 0:
            return max(floor, self._expected)
        r = (measured / self._sr) / elapsed
        if r >= 1.0:
            return floor
        remaining = max(0, self._expected - self._seg_written)
        deficit = int((1.0 - r) / max(r, self._RATE_FLOOR) * remaining)
        if deficit <= 0:
            return floor
        return max(floor, deficit + int(self._MARGIN_SECS * self._sr))

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
            self._seg_written += chunk.shape[0]
            self._buf.append(chunk)
            self._buffered += chunk.shape[0]
            if not self._armed and self._buffered >= self._arm_target(now):
                self._armed = True
                self._fire_audible()
            self._cv.notify_all()

    def flush(self, epoch: int) -> None:
        """Block until buffered audio for `epoch` has fully played (or the epoch is cancelled)."""
        with self._cv:
            # Short utterances may never reach the arm target; and while draining, an
            # empty buffer means "done", so the callback must not treat it as a stall.
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
        """Instant stop for barge-in: drop the buffer and invalidate the current epoch."""
        with self._cv:
            self._epoch += 1
            self._buf.clear()
            self._head = self._buffered = 0
            self._armed = False
            self._on_audible = None
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
            # Streaming: write each chunk the moment it's produced; pace() tells the sink
            # how much audio this sentence should yield so it holds back just enough lead.
            for sentence in sentences:
                if self._gen_id != my_gen:
                    break
                self._sink.pace(estimate_speech_secs(sentence))
                with self.synth_lock:
                    for chunk in self._iter_sentence(sentence, my_gen):
                        self._sink.write(chunk, my_gen)
        self._sink.flush(my_gen)

    def speak_stream(self, text_stream: Iterator[str]) -> str:
        """Speak a reply while it is still being generated (thinking ∥ speaking).

        Sentences are consumed from `text_stream` as they complete, so the first
        sentence's audio starts long before the LLM finishes the reply. Each sentence's
        audio is paced by the sink's lead-aware buffer, so a sub-realtime source starts
        with just enough lead to play through smoothly; any wait for the next sentence
        lands on a sentence boundary, where a beat of silence reads as a natural pause.

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
            with self.synth_lock:
                for chunk in self._iter_sentence(sentence, my_gen):
                    self._sink.write(chunk, my_gen)
        self._sink.flush(my_gen)
        return " ".join(spoken)
