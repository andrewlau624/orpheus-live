"""Tests for TTS backend selection and the remote client's PCM reassembly + wire contract."""

import numpy as np
import pytest

from orpheus_live.config import Settings
from orpheus_live.engines.factory import resolve_backend
from orpheus_live.engines.tts_remote import RemoteOrpheusVoice, pcm_frames
from orpheus_live.models import Voice, VoicePreset

httpx = pytest.importorskip("httpx")  # ships transitively via ollama; skip if absent


def _preset() -> VoicePreset:
    return VoicePreset(
        name="test", voice=Voice.TARA, temperature=0.6, top_p=0.9, repetition_penalty=1.2
    )


# -- backend selection --------------------------------------------------------


def test_auto_and_mlx_resolve_to_local():
    assert resolve_backend(Settings(tts_backend="auto")) == "mlx"
    assert resolve_backend(Settings(tts_backend="mlx")) == "mlx"


def test_remote_is_explicit_opt_in():
    assert resolve_backend(Settings(tts_backend="remote")) == "remote"


def test_cpp_is_explicit_opt_in():
    assert resolve_backend(Settings(tts_backend="cpp")) == "cpp"


# -- PCM reassembly across arbitrary chunk boundaries -------------------------


def test_pcm_frames_reassembles_samples_split_across_chunks():
    samples = np.array([0.0, 0.25, -0.5, 0.75, 1.0], dtype="<f4")
    raw = samples.tobytes()  # 20 bytes
    # Deliberately split mid-sample (odd byte offsets) to exercise the remainder buffer.
    chunks = [raw[:3], raw[3:9], raw[9:10], raw[10:]]
    out = np.concatenate(list(pcm_frames(chunks)))
    np.testing.assert_array_equal(out, samples.astype(np.float32))


def test_pcm_frames_drops_trailing_partial_sample():
    samples = np.array([1.0, 2.0], dtype="<f4")
    raw = samples.tobytes() + b"\x00\x01"  # two extra bytes = truncated sample
    out = np.concatenate(list(pcm_frames([raw])))
    np.testing.assert_array_equal(out, samples.astype(np.float32))


def test_pcm_frames_handles_empty_and_zero_length_chunks():
    assert list(pcm_frames([])) == []
    assert list(pcm_frames([b"", b""])) == []


def test_pcm_frames_batches_small_yields_into_larger_chunks():
    # Each chunk is 4 bytes (one sample) — should accumulate before yielding.
    one_sample = np.array([1.0], dtype="<f4").tobytes()
    # 100 single-sample chunks = 100 samples total, but should yield in ~480-sample batches
    chunks = [one_sample] * 100
    result = list(pcm_frames(chunks))
    # With only 100 samples (< 480 threshold), should yield once at the end
    assert len(result) == 1
    assert result[0].shape[0] == 100


# -- remote client end-to-end against a fake server ---------------------------


def _mock_client(handler) -> "httpx.Client":
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")


def test_synthesize_posts_preset_and_returns_audio():
    seen = {}
    samples = np.array([0.1, 0.2, 0.3, 0.4], dtype="<f4")

    def handler(request: "httpx.Request") -> "httpx.Response":
        import json

        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=samples.tobytes())

    voice = RemoteOrpheusVoice(Settings(), _preset(), client=_mock_client(handler))
    out = voice.synthesize("hello there")

    assert seen["url"].endswith("/tts")
    assert seen["body"] == {
        "text": "hello there",
        "voice": "tara",
        "temperature": 0.6,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
    }
    np.testing.assert_array_equal(out, samples.astype(np.float32))


def test_stream_raises_on_server_error():
    def handler(request: "httpx.Request") -> "httpx.Response":
        return httpx.Response(500, content=b"boom")

    voice = RemoteOrpheusVoice(Settings(), _preset(), client=_mock_client(handler))
    with pytest.raises(httpx.HTTPStatusError):
        list(voice.stream("hi"))
