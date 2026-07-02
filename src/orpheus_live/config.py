"""Runtime configuration for Orpheus Live."""

import os
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ORPHEUS_LIVE_")

    # Mic / VAD. VAD is only a cheap speech-presence gate now; the cognition LLM makes the
    # turn-taking decision (respond / wait / interrupt / backchannel) on each pause.
    mic_sample_rate: int = 16000  # mic capture rate (Whisper + VAD friendly)
    frame_ms: int = 32  # silero-vad requires exactly 512 samples/chunk @16k = 32ms
    vad_threshold: float = 0.5  # speech probability (0..1) above which a frame counts as speech
    vad_threshold_during_ai_speech: float = 0.85  # when AI speaks, raise threshold to avoid
    # picking up its own voice through speakers (acoustic echo suppression).
    start_speech_ms: int = 180  # voiced audio needed to *start* a turn
    min_utterance_ms: int = 350  # ignore blips shorter than this before judging a pause
    post_speak_cooldown: float = 0.35  # ignore mic briefly after the AI stops talking (echo guard)
    # Lag-aware pickup: once the AI commits to a reply, mute the mic until its first audio is
    # actually audible (LLM generate + TTS synth = ~1-2s). Stops the "I finished, now I wait"
    # gap from picking up breath/keyboard/room noise and spinning up extra cognition.
    lag_aware: bool = True
    # Turn pacing. A short pause asks cognition "are they done?"; a long pause forces a turn
    # end so a turn can never hang if the model keeps saying "wait".
    turn_pause_ms: int = 500  # silence after speech -> consult cognition about the turn
    turn_end_ms: int = 900  # silence this long -> force the turn to end (safety net)

    # STT
    whisper_repo: str = "mlx-community/whisper-large-v3-turbo"
    stt_language: str = "en"  # force language (avoids garbage from language misdetection)
    stt_min_rms: float = 0.006  # clips quieter than this are treated as non-speech (no transcribe)

    # TTS backend selection. "mlx" runs Orpheus locally on Apple Silicon (default);
    # "remote" offloads synthesis to an Orpheus server (e.g. vLLM + SNAC on an NVIDIA
    # GPU) over HTTP for much lower latency; "auto" picks mlx locally. STT/LLM stay
    # local either way -- only TTS is offloaded.
    tts_backend: Literal["auto", "mlx", "remote", "cpp"] = "auto"
    tts_remote_url: str = "http://localhost:8000"  # base URL of the Orpheus TTS server
    tts_remote_timeout_s: float = 30.0  # per-request timeout for the remote backend
    # "cpp" backend: Orpheus via orpheus-cpp (llama.cpp GGUF + ONNX SNAC). Local, and on
    # Apple Silicon it offloads to Metal -- may beat the mlx path (needs the llama-cpp-python
    # metal wheel; see README). n_gpu_layers=-1 offloads all layers to the GPU.
    orpheus_cpp_lang: str = "en"
    orpheus_cpp_n_gpu_layers: int = -1

    # TTS (Orpheus via mlx-audio; model auto-downloads to the HF cache)
    orpheus_model: str = "mlx-community/orpheus-3b-0.1-ft-4bit"
    tts_sample_rate: int = 24000
    # Playback mode. Default (False) buffers each sentence fully before playing it: smooth
    # on Apple Silicon, where Orpheus generates slower than realtime, so playing chunks as
    # they arrive underruns into choppy stop-and-go. Latency is hidden instead by speculation
    # (pre-synth during your speech) + sentence pipelining. Set True only on faster-than-
    # realtime hardware (or the remote GPU backend), where live chunks lower TTFB without
    # underrunning. Seams are correct either way (overlap-save decode).
    tts_streaming: bool = True
    tts_first_chunk_frames: int = 3  # tiny first chunk -> first audio out ASAP (streaming mode)
    tts_chunk_frames: int = 12  # larger later chunks -> fewer decode calls once flowing
    tts_context_frames: int = 4  # overlap carried between chunks for seamless seams
    tts_prebuffer_s: float = 0.2  # buffer this much before playback starts (rides out lag spikes)
    # Output-stream block size (samples). Larger = the audio callback fires less often with a
    # looser deadline, so it survives GIL stalls / CPU thrash without crackling; costs a little
    # latency (2048 @ 24kHz ~= 85ms). Bump to 4096 if you still hear crackle under load.
    tts_output_blocksize: int = 2048
    saved_voices_dir: str = "saved_voices"  # type a name + Enter to save the voice here

    # Every launch without a named preset randomizes the voice and delivery,
    # then keeps them fixed for the whole session. Ranges stay in Orpheus's
    # stable zone: temp below ~0.55 or repetition_penalty near the 1.1 floor
    # makes it loop/repeat phrases.
    rand_temp: tuple[float, float] = (0.55, 0.8)
    rand_top_p: tuple[float, float] = (0.85, 0.95)
    rand_rep_penalty: tuple[float, float] = (1.15, 1.35)

    # LLM
    ollama_model: str = "llama3.2:3b"

    # Cognition (silence self-questioning) -- a dedicated small/fast model so
    # cognition ticks never compete with reply generation.
    cognition_model: str = "llama3.2:1b"
    cognition_tick_s: float = 0.4  # how often the background thread checks in
    cognition_base_silence_s: float = 2.5  # silence before the first consult
    cognition_jitter_frac: float = 0.4  # +/- fraction applied to every check interval
    # After the AI finishes talking, the next beat belongs to the user -- that pause is
    # them digesting/formulating, not awkward silence. Cognition only starts counting
    # silence after this grace window.
    cognition_post_ai_grace_s: float = 2.0

    # Thinking-while-listening: transcribe the in-progress utterance every
    # interval and speculatively start generating a reply, so it's ~ready the
    # moment the user stops talking.
    speculate: bool = True
    speculation_interval_s: float = 0.8  # transcribe partials this often (kept clear of the
    # decision path's own transcribe, which shares the transcriber lock; large-v3-turbo is
    # slower than base, so a tight interval here would starve the reply-latency-critical path)
    speculation_take_timeout_s: float = 10.0  # max wait for an in-flight matching reply

    @property
    def frame_len(self) -> int:
        return self.mic_sample_rate * self.frame_ms // 1000


def configure_hf_token() -> None:
    """Use the local HF token (nicer download rate limits) if present."""
    tok_path = Path("~/.cache/huggingface/token").expanduser()
    if tok_path.exists() and "HF_TOKEN" not in os.environ:
        os.environ["HF_TOKEN"] = tok_path.read_text().strip()


settings = Settings()
