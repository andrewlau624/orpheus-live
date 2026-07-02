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
    """

    _FADE = 240  # samples of fade-in at the start of an utterance (declick)

    def __init__(self, settings: Settings):
        self._sr = settings.tts_sample_rate
        self._prebuffer = int(settings.tts_prebuffer_s * self._sr)
        self._max_buffered = int(2.0 * self._sr)  # backpressure cap (~2s ahead)
        self._buf: deque[np.ndarray] = deque()
        self._head = 0  # read cursor into buf[0]
        self._buffered = 0  # total samples across buf
        self._epoch = 0
        self._armed = False  # gate playback until prebuffer is met
        self._faded = False  # fade-in applied to this utterance yet?
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
            self._cv.notify_all()

    def begin(self, epoch: int) -> None:
        """Open a new utterance: adopt `epoch`, drop any leftover audio, re-arm on prebuffer."""
        with self._cv:
            self._epoch = epoch
            self._buf.clear()
            self._head = self._buffered = 0
            self._armed = False
            self._faded = False
            self._cv.notify_all()

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
            self._buf.append(chunk)
            self._buffered += chunk.shape[0]
            if self._buffered >= self._prebuffer:
                self._armed = True
            self._cv.notify_all()

    def flush(self, epoch: int) -> None:
        """Block until buffered audio for `epoch` has fully played (or the epoch is cancelled)."""
        with self._cv:
            self._armed = True  # short utterances may never reach the prebuffer target
            self._cv.notify_all()
            while self._buffered > 0 and epoch == self._epoch:
                self._cv.wait()

    def clear(self) -> None:
        """Instant stop for barge-in: drop the buffer and invalidate the current epoch."""
        with self._cv:
            self._epoch += 1
            self._buf.clear()
            self._head = self._buffered = 0
            self._armed = False
            self._cv.notify_all()

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


class SpeechPlayer:
    """Synthesizes and plays a reply, cancellable mid-utterance.

    Two modes (chosen by `buffer_whole`):

    - buffered (default on sub-realtime HW): synthesize the ENTIRE reply first, then play
      it. Nothing generates while audio plays, so the real-time audio callback is never
      starved by MLX work on the GIL -- no periodic choppiness, no inter-sentence gaps.
    - streaming: write each sentence's chunks to the sink as they're produced (lower TTFB,
      only smooth when generation outpaces playback, e.g. a GPU backend).

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
        # Fired once per speak(), the moment the first audio reaches the sink -- used by the
        # orchestrator to un-mute the mic (lag-aware pickup) exactly when speech becomes audible.
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

    def _write(self, buf: np.ndarray, my_gen: int, state: list[bool]) -> None:
        """Write to the sink, firing on_first_audio exactly once (on the first buffer)."""
        if not state[0]:
            state[0] = True
            if self._on_first_audio is not None:
                self._on_first_audio()
        self._sink.write(buf, my_gen)

    def speak(self, text: str) -> None:
        """Synthesize and play `text`; bails early if cancel() runs concurrently."""
        my_gen = self._gen_id
        self._sink.begin(my_gen)
        sentences = split_sentences(text)
        fired = [False]  # one-shot latch for on_first_audio, threaded through _write
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
                self._write(clip, my_gen, fired)
        else:
            # Streaming: write each chunk to the sink the moment it's produced (low TTFB).
            for sentence in sentences:
                if self._gen_id != my_gen:
                    break
                with self.synth_lock:
                    for chunk in self._iter_sentence(sentence, my_gen):
                        self._write(chunk, my_gen, fired)
        self._sink.flush(my_gen)

    def speak_stream(self, text_stream: Iterator[str]) -> str:
        """Speak a reply while it is still being generated (thinking ∥ speaking).

        Sentences are consumed from `text_stream` as they complete, so the first
        sentence's audio starts long before the LLM finishes the reply. Each sentence
        is still synthesized to a full clip before it plays (smooth on sub-realtime
        hardware); any wait for the next sentence lands on a sentence boundary, where
        a beat of silence reads as a natural pause rather than mid-word chop.

        Returns the full reply text (everything actually generated), even if playback
        was cancelled partway, so the caller can still log/remember it.
        """
        my_gen = self._gen_id
        self._sink.begin(my_gen)
        fired = [False]
        spoken: list[str] = []
        for sentence in iter_stream_sentences(text_stream):
            spoken.append(sentence)  # collect even when cancelled: memory wants the text
            if self._gen_id != my_gen:
                continue  # cancelled: keep draining the LLM cheaply, synthesize nothing
            with self.synth_lock:
                for chunk in self._iter_sentence(sentence, my_gen):
                    self._write(chunk, my_gen, fired)
        self._sink.flush(my_gen)
        return " ".join(spoken)
