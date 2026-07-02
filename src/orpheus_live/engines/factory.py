"""Backend selection: build the right TTS engine (mlx / remote GPU / llama.cpp) from settings."""

from ..config import Settings
from ..models import VoicePreset
from .base import TtsBackend


def resolve_backend(settings: Settings) -> str:
    """Resolve `tts_backend` to a concrete backend name ("mlx" | "remote" | "cpp").

    Pure and side-effect-free (no imports of heavy deps) so it's easy to test. "auto"
    picks the local MLX engine; "remote" and "cpp" are explicit opt-ins.
    """
    if settings.tts_backend in ("remote", "cpp"):
        return settings.tts_backend
    return "mlx"  # "auto" and "mlx" both run locally on MLX


def load_voice(settings: Settings, preset: VoicePreset) -> TtsBackend:
    """Construct the configured TTS backend. Heavy deps are imported lazily per backend."""
    backend = resolve_backend(settings)
    if backend == "remote":
        from .tts_remote import RemoteOrpheusVoice

        return RemoteOrpheusVoice(settings, preset)
    if backend == "cpp":
        from .tts_cpp import OrpheusCppVoice

        return OrpheusCppVoice(settings, preset)
    from .tts import OrpheusVoice

    return OrpheusVoice(settings, preset)
