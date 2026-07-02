"""Local TTS backend via orpheus-cpp (llama.cpp GGUF + ONNX SNAC), CPU or Apple-Silicon Metal.

An alternative to the MLX backend that runs Orpheus through llama.cpp. On Apple Silicon it
can offload to Metal (`orpheus_cpp_n_gpu_layers`), which may generate faster than the MLX
path -- and faster-than-realtime generation is exactly what makes streaming playback smooth.
Conforms to `TtsBackend`, so `tts_backend="cpp"` swaps it in with no other changes.

Requires the optional deps: `pip install "orpheus-live[cpp]"` plus a llama-cpp-python wheel
(Metal on Apple Silicon): see the README. Presets map voice/temperature/top_p; Orpheus-cpp
doesn't expose repetition_penalty, so that preset field is ignored by this backend.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..config import Settings
from ..console import AI, DIM, log
from ..models import VoicePreset
from .tts import save_preset


def _to_float32(samples) -> np.ndarray:
    """orpheus-cpp yields int16 PCM; the rest of the pipeline is float32 in [-1, 1]."""
    return np.asarray(samples, dtype=np.int16).reshape(-1).astype(np.float32) / 32768.0


class OrpheusCppVoice:
    """The orpheus-cpp engine plus this session's fixed voice preset (conforms to TtsBackend)."""

    def __init__(self, settings: Settings, preset: VoicePreset):
        from orpheus_cpp import OrpheusCpp  # lazy: only needed for this backend

        self.settings = settings
        self.preset = preset
        log(f"  · loading Orpheus (orpheus-cpp, {settings.orpheus_cpp_lang})...", DIM)
        self._model = OrpheusCpp(
            lang=settings.orpheus_cpp_lang,
            n_gpu_layers=settings.orpheus_cpp_n_gpu_layers,
            verbose=False,
        )

    def _options(self) -> dict:
        p = self.preset
        # orpheus-cpp accepts voice_id/temperature/top_p (no repetition_penalty).
        return {"voice_id": str(p.voice), "temperature": p.temperature, "top_p": p.top_p}

    def stream(self, text: str) -> Iterator[np.ndarray]:
        for _sr, chunk in self._model.stream_tts_sync(text, options=self._options()):
            arr = _to_float32(chunk)
            if arr.size:
                yield arr

    def synthesize(self, text: str) -> np.ndarray:
        _sr, samples = self._model.tts(text, options=self._options())
        return _to_float32(samples)

    def save_as(self, name: str) -> Path:
        self.preset = self.preset.model_copy(update={"name": name})
        path = save_preset(self.settings, self.preset)
        log(f"\n  ★ saved this voice as '{name}' -> {path}", AI)
        log(f"  run it again anytime: make run {name}", DIM)
        return path
