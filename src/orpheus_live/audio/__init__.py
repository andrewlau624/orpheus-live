"""Audio I/O: mic capture + VAD segmentation, playback, and voice-activity detection."""

from .capture import AudioIn, MuteGate
from .playback import (
    AudioSink,
    PreSynthStream,
    SpeechPlayer,
    iter_stream_sentences,
    split_sentences,
)
from .vad import Vad

__all__ = [
    "AudioIn",
    "MuteGate",
    "AudioSink",
    "PreSynthStream",
    "SpeechPlayer",
    "Vad",
    "iter_stream_sentences",
    "split_sentences",
]
