"""Voice-activity-detection wrapper, backed by silero-vad.

Silero is a stateful/recurrent model: its internal context is only valid if
every sequential audio chunk is fed to it in order, so `probability`/
`is_speech` must be called on each mic frame in sequence (never skipped or
reordered, and never called twice for the same frame) -- which is exactly how
`audio_io.py`'s single capture loop already drives it, for both normal
utterance segmentation and barge-in detection. It returns a continuous speech
*probability* per chunk rather than webrtcvad's binary decision, much more
robust to breath/background noise; `is_speech` thresholds it for callers that
just want a boolean.
"""

import numpy as np
import torch
from silero_vad import load_silero_vad


class Vad:
    def __init__(self, threshold: float, sample_rate: int):
        torch.set_num_threads(1)
        self._model = load_silero_vad()
        self._threshold = threshold
        self._sample_rate = sample_rate

    def probability(self, frame: bytes, threshold: float | None = None) -> float:
        try:
            pcm = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
            chunk = torch.from_numpy(pcm).unsqueeze(0)
            return self._model(chunk, self._sample_rate).item()
        except Exception:
            return 0.0

    def is_speech(self, frame: bytes, threshold: float | None = None) -> bool:
        return self.probability(frame) >= (threshold if threshold is not None else self._threshold)
