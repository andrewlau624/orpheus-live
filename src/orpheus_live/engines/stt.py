"""mlx-whisper speech-to-text wrapper, hardened against non-speech hallucination.

Whisper happily invents text for non-speech audio (keyboard clicks, fan noise) and can
loop a token -- "rafa rafa rafa rafa". Three guards keep that out: an RMS energy gate
(don't transcribe near-silence), Whisper's own no-speech / low-confidence thresholds with
`condition_on_previous_text=False` (kills the repetition loops), and a post-filter that
drops output dominated by one repeated token.
"""

import threading
from collections import Counter

import mlx_whisper
import numpy as np

from ..config import Settings


def _looks_like_loop(text: str) -> bool:
    """True if `text` is a repetition hallucination (one token dominating a long run).

    Whisper's "rafa rafa rafa..." loops are *long* runs of one token. A person really
    saying "hello hello hello" or "no no no" is short, so the bar is high (6+ tokens,
    80%+ dominated by one word) -- otherwise this guard eats genuine repeated speech.
    """
    words = text.lower().split()
    if len(words) < 6:
        return False  # too short to be a confident loop; let real repeats through
    _, count = Counter(words).most_common(1)[0]
    return count / len(words) >= 0.8


class Transcriber:
    def __init__(self, settings: Settings):
        self.settings = settings
        # Serialize calls: the speculation tick and the main loop can both ask
        # for a transcription near-simultaneously, and mlx-whisper's decoding
        # state isn't guaranteed thread-safe.
        self._lock = threading.Lock()

    def warm_up(self) -> None:
        self.transcribe(np.zeros(self.settings.mic_sample_rate, dtype=np.float32))

    def _too_quiet(self, audio: np.ndarray) -> bool:
        if audio.size == 0:
            return True
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        return rms < self.settings.stt_min_rms

    def transcribe(self, audio: np.ndarray) -> str:
        if self._too_quiet(audio):
            return ""  # silence / faint noise -> nothing worth decoding
        with self._lock:
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.settings.whisper_repo,
                language=self.settings.stt_language,
                temperature=0.0,  # deterministic; less prone to inventing text
                condition_on_previous_text=False,  # stops "rafa rafa rafa" loops
                compression_ratio_threshold=2.4,  # reject over-repetitive decodes
                logprob_threshold=-1.0,  # reject low-confidence decodes
                no_speech_threshold=0.6,  # treat no-speech segments as empty
            )
        text = result["text"].strip()
        return "" if _looks_like_loop(text) else text
