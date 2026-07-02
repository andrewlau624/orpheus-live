"""Orpheus-TTS engine (via mlx-audio), voice presets, and the save-a-voice workflow.

Orpheus has no speaker embeddings -- a "voice" is one of eight named base
voices plus delivery parameters (temperature/top_p/repetition_penalty). A
launch without a preset rolls a random combination and keeps it fixed for the
session; typing a name + Enter saves it as `saved_voices/<Name>.json` for
`orpheus-live <name>` / `make run <name>` to load later.

Audio is produced by a *correct* incremental SNAC decode (see `OrpheusVoice.stream`).
mlx-audio 0.4.4's own streaming decode trims a fixed `context_frames * hop_length`
window that doesn't match the few context frames it actually prepends, so its chunk
seams audibly repeat a beat ("h-h-h-hi"). We decode each chunk with left-context
overlap-save and trim exactly the context region, which is seamless.
"""

import json
import random
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..config import Settings
from ..console import AI, DIM, log
from ..models import ORPHEUS_VOICES, VoicePreset

if TYPE_CHECKING:
    from .base import TtsBackend

_CODES_PER_FRAME = 7  # Orpheus emits 7 SNAC codes per audio frame
_START_OF_AUDIO = 128257  # SOA: audio codes follow this generated token
_END_OF_SPEECH = 128258
_CODE_OFFSET = 128266  # audio token value - this = SNAC code
_REPETITION_CONTEXT = 64  # lib default 20 (~0.15s) misses phrase loops


def decode_window(
    flat: list,
    emitted: int,
    upto: int,
    context_frames: int,
    decode: Callable[[list], np.ndarray],
    samples_per_frame: int | None,
) -> tuple[np.ndarray, int | None]:
    """Decode new SNAC frames [emitted, upto) with left-context overlap-save.

    `context_frames` preceding frames are prepended to warm the decoder's receptive
    field, then trimmed off using the *measured* samples-per-frame -- so exactly the
    context region is removed, with no repeated or dropped audio at the seam. (This is
    the bug in mlx-audio 0.4.4's own streaming decode, which trims a fixed
    `context_frames * hop_length` that doesn't match what it prepends.)

    Returns the seamless new-frame audio and the (now-known) samples-per-frame.
    """
    start = max(0, emitted - context_frames)
    window = flat[start * _CODES_PER_FRAME : upto * _CODES_PER_FRAME]
    if not window:
        return np.zeros(0, dtype=np.float32), samples_per_frame
    audio = decode(window)
    n_frames = len(window) // _CODES_PER_FRAME
    if samples_per_frame is None and n_frames > 0:
        samples_per_frame = audio.shape[0] // n_frames
    trim = (emitted - start) * (samples_per_frame or 0)
    return audio[trim:], samples_per_frame


def random_preset(settings: Settings) -> VoicePreset:
    """Roll a fresh random voice + delivery for this session (fixed once chosen)."""
    return VoicePreset(
        name="random",
        voice=random.choice(ORPHEUS_VOICES),
        temperature=round(random.uniform(*settings.rand_temp), 2),
        top_p=round(random.uniform(*settings.rand_top_p), 2),
        repetition_penalty=round(random.uniform(*settings.rand_rep_penalty), 2),
    )


def preset_path(settings: Settings, name: str) -> Path:
    return Path(settings.saved_voices_dir) / f"{name}.json"


def list_presets(settings: Settings) -> list[str]:
    d = Path(settings.saved_voices_dir)
    if not d.is_dir():
        return []
    return sorted(f.stem for f in d.glob("*.json"))


def load_preset(settings: Settings, name: str) -> VoicePreset:
    """Load a saved preset by name (case-insensitive); raises with help if missing."""
    for existing in list_presets(settings):
        if existing.lower() == name.lower():
            return VoicePreset.model_validate_json(preset_path(settings, existing).read_text())
    available = ", ".join(list_presets(settings)) or "(none saved yet)"
    raise SystemExit(f"No saved voice named '{name}'. Available: {available}")


def save_preset(settings: Settings, preset: VoicePreset) -> Path:
    Path(settings.saved_voices_dir).mkdir(parents=True, exist_ok=True)
    path = preset_path(settings, preset.name)
    path.write_text(json.dumps(preset.model_dump(), indent=2) + "\n")
    return path


class OrpheusVoice:
    """The Orpheus engine plus this session's (fixed) voice preset.

    Not thread-safe on its own -- MLX runs one generation at a time. Callers
    serialize `stream`/`synthesize` (the SpeechPlayer holds a single synth lock).
    """

    def __init__(self, settings: Settings, preset: VoicePreset):
        from mlx_audio.tts.utils import load_model

        self.settings = settings
        self.preset = preset
        log(f"  · loading Orpheus ({settings.orpheus_model})...", DIM)
        self._model = load_model(settings.orpheus_model)

    def _max_tokens(self, text: str) -> int:
        # Short prompts make Orpheus ramble/repeat until the budget runs out
        # (~137.5 tokens/sec of audio, a spoken word ~55 tokens), so scale it.
        return min(1200, 160 + 80 * max(1, len(text.split())))

    def _decode(self, codes: list) -> np.ndarray:
        """Decode a slice of flat SNAC codes (a multiple of 7) to a mono waveform."""
        from mlx_audio.tts.models.llama.llama import decode_audio_from_codes

        return np.asarray(decode_audio_from_codes(codes)[0], dtype=np.float32).reshape(-1)

    def stream(self, text: str) -> Iterator[np.ndarray]:
        """Yield seamless float32 audio chunks @24kHz as tokens are generated.

        Each chunk covers `tts_chunk_frames` new SNAC frames, decoded together with
        `tts_context_frames` of preceding frames as left-context warmup; the context
        samples are then trimmed off using the *measured* samples-per-frame, so the trim
        exactly matches what was prepended -- no repeated/dropped audio at the seams.

        SNAC codes are accumulated incrementally as tokens arrive (audio codes are the
        generated tokens after the start-of-audio marker, minus `_CODE_OFFSET`). This
        avoids re-parsing the whole token tensor every frame, which otherwise hogs the
        GIL and can starve the audio callback into crackles on a loaded machine.
        """
        import mlx.core as mx
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        p = self.preset
        s = self.settings
        chunk = max(1, s.tts_chunk_frames)
        first_chunk = max(1, s.tts_first_chunk_frames)
        ctx = max(0, s.tts_context_frames)

        input_ids = self._model.prepare_input_ids(text, p.voice)
        sampler = make_sampler(p.temperature, p.top_p)
        logits_processors = make_logits_processors(None, p.repetition_penalty, _REPETITION_CONTEXT)

        codes: list[int] = []  # flat SNAC codes so far (offset-normalized)
        in_audio = False  # have we passed the start-of-audio marker?
        emitted = 0  # frames already yielded
        spf: int | None = None  # samples per frame, measured on first decode

        def emit(upto: int) -> Iterator[np.ndarray]:
            """Decode and yield new frames [emitted, upto) with left-context warmup."""
            nonlocal emitted, spf
            if upto <= emitted:
                return
            out, spf = decode_window(codes, emitted, upto, ctx, self._decode, spf)
            emitted = upto
            if out.size:
                yield out

        for i, response in enumerate(
            stream_generate(
                self._model,
                tokenizer=self._model.tokenizer,
                prompt=input_ids.squeeze(0),
                max_tokens=self._max_tokens(text),
                sampler=sampler,
                logits_processors=logits_processors,
            )
        ):
            if i % 50 == 0:
                mx.clear_cache()
            token = int(response.token)
            if token == _END_OF_SPEECH:
                break
            if token == _START_OF_AUDIO:
                in_audio = True  # audio codes start after this marker
                codes.clear()  # keep only codes after the last SOA (matches parse_output)
                continue
            if not in_audio:
                continue
            codes.append(token - _CODE_OFFSET)
            # Emit a tiny first chunk for low TTFB, then larger chunks once audio flows.
            total = len(codes) // _CODES_PER_FRAME
            while total - emitted >= (first_chunk if emitted == 0 else chunk):
                yield from emit(emitted + (first_chunk if emitted == 0 else chunk))

        # Flush every remaining frame in one final window (no look-ahead needed).
        yield from emit(len(codes) // _CODES_PER_FRAME)
        mx.clear_cache()

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize `text` to one complete float32 clip @24kHz in a single clean decode.

        This is the default playback path: on this hardware Orpheus generates slower than
        realtime, so a whole sentence is decoded at once (one seamless SNAC pass) and then
        played, rather than streamed chunk-by-chunk (which would underrun into silence).
        Latency is hidden by speculation + sentence pipelining, not by within-sentence
        streaming. Use `stream()` (opt-in via `tts_streaming`) only on faster-than-realtime
        hardware.
        """
        p = self.preset
        chunks = [
            np.asarray(result.audio, dtype=np.float32).reshape(-1)
            for result in self._model.generate(
                text=text,
                voice=p.voice,
                temperature=p.temperature,
                top_p=p.top_p,
                repetition_penalty=p.repetition_penalty,
                repetition_context_size=_REPETITION_CONTEXT,
                max_tokens=self._max_tokens(text),
                verbose=False,
            )
        ]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def save_as(self, name: str) -> Path:
        """Persist the session's voice under `name` for later `orpheus-live <name>`."""
        self.preset = self.preset.model_copy(update={"name": name})
        path = save_preset(self.settings, self.preset)
        log(f"\n  ★ saved this voice as '{name}' -> {path}", AI)
        log(f"  run it again anytime: make run {name}", DIM)
        return path


def start_save_listener(voice: "TtsBackend") -> threading.Thread:
    """Background thread: type a name + Enter anytime to save the current voice."""

    def _listen() -> None:
        while True:
            try:
                line = input()
            except (EOFError, OSError):
                return
            name = line.strip()
            if name and name.isprintable() and "/" not in name:
                voice.save_as(name)

    t = threading.Thread(target=_listen, daemon=True)
    t.start()
    return t
