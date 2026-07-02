"""Remote TTS backend: offload Orpheus synthesis to an HTTP server (e.g. vLLM + SNAC on a GPU).

The server exposes `POST /tts` taking `{text, voice, temperature, top_p, repetition_penalty}`
and streaming back raw little-endian float32 PCM @ 24kHz mono. This client yields those
samples as numpy chunks the moment they arrive, so the persistent AudioSink plays remote
audio exactly like local audio. See `server/orpheus_server.py` for a reference server and
the wire contract. STT and the LLM stay local; only synthesis is offloaded.
"""

from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from ..config import Settings
from ..console import AI, DIM, log
from ..models import VoicePreset
from .tts import save_preset

_PCM_DTYPE = np.dtype("<f4")  # little-endian float32, the server's wire format
_SAMPLE_BYTES = _PCM_DTYPE.itemsize


def pcm_frames(byte_chunks: Iterable[bytes]) -> Iterator[np.ndarray]:
    """Reassemble a byte stream into float32 sample arrays, honoring 4-byte boundaries.

    Network chunks split anywhere, so a float32 sample may straddle two chunks; we buffer
    the remainder and only emit whole samples. Pure/synchronous so it's unit-testable.
    """
    remainder = b""
    for raw in byte_chunks:
        if not raw:
            continue
        buf = remainder + raw
        n = len(buf) - (len(buf) % _SAMPLE_BYTES)
        if n:
            yield np.frombuffer(buf[:n], dtype=_PCM_DTYPE).astype(np.float32)
        remainder = buf[n:]
    # Any trailing < 4 bytes is a truncated sample -> drop it.


class RemoteOrpheusVoice:
    """Streams synthesis from a remote Orpheus server; conforms to `TtsBackend`.

    `client` may be injected for testing (an httpx.Client with a MockTransport); in
    production one is built from `settings.tts_remote_url`.
    """

    def __init__(self, settings: Settings, preset: VoicePreset, client=None):
        self.settings = settings
        self.preset = preset
        if client is None:
            import httpx  # lazy: only needed for the remote backend

            client = httpx.Client(
                base_url=settings.tts_remote_url, timeout=settings.tts_remote_timeout_s
            )
        self._client = client
        log(f"  · using remote TTS at {settings.tts_remote_url}", DIM)

    def _payload(self, text: str) -> dict:
        p = self.preset
        return {
            "text": text,
            "voice": str(p.voice),
            "temperature": p.temperature,
            "top_p": p.top_p,
            "repetition_penalty": p.repetition_penalty,
        }

    def stream(self, text: str) -> Iterator[np.ndarray]:
        with self._client.stream("POST", "/tts", json=self._payload(text)) as response:
            response.raise_for_status()
            yield from pcm_frames(response.iter_bytes())

    def synthesize(self, text: str) -> np.ndarray:
        chunks = list(self.stream(text))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def save_as(self, name: str) -> Path:
        """Persist this session's voice preset under `name` (presets are backend-agnostic)."""
        self.preset = self.preset.model_copy(update={"name": name})
        path = save_preset(self.settings, self.preset)
        log(f"\n  ★ saved this voice as '{name}' -> {path}", AI)
        log(f"  run it again anytime: make run {name}", DIM)
        return path
