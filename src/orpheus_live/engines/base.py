"""The TTS backend interface: any voice engine the conversation loop can drive.

Two implementations conform to it: `OrpheusVoice` (local MLX, Apple Silicon) and
`RemoteOrpheusVoice` (offloads synthesis to an Orpheus HTTP server, e.g. vLLM + SNAC
on an NVIDIA GPU). `core.conversation` builds one via `engines.load_voice` and only
ever calls this surface, so backends are interchangeable.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..models import VoicePreset


@runtime_checkable
class TtsBackend(Protocol):
    """A synthesizer for one fixed voice preset."""

    preset: VoicePreset

    def stream(self, text: str) -> Iterator[np.ndarray]:
        """Yield float32 mono audio chunks @ the configured sample rate as they're ready."""
        ...

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize `text` to one complete float32 mono clip."""
        ...

    def save_as(self, name: str) -> Path:
        """Persist this session's voice preset under `name`."""
        ...
