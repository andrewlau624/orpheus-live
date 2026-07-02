"""Tests for the seamless overlap-save decode that replaces mlx-audio's glitchy streaming.

The bug being guarded against: mlx-audio 0.4.4 trims a fixed `context_frames * hop_length`
window that doesn't match the few context frames it actually prepends, so chunk seams
repeat/drop a beat of audio ("h-h-h-hi"). `decode_window` trims exactly the context region
using the measured samples-per-frame, so streamed chunks reconstruct the audio bit-for-bit.
"""

import numpy as np

from orpheus_live.engines.tts import decode_window

_SPF = 4  # fake samples per SNAC frame
_CODES = 7


def _flat(n_frames: int) -> list[int]:
    """A flat code list where frame f's first code is f (so audio can be traced to frames)."""
    codes: list[int] = []
    for f in range(n_frames):
        codes += [f, 0, 0, 0, 0, 0, 0]
    return codes


def _fake_decode(codes: list) -> np.ndarray:
    """Decode = each frame -> _SPF samples all equal to that frame's id (codes[f*7])."""
    n = len(codes) // _CODES
    return np.array(
        [float(codes[f * _CODES]) for f in range(n) for _ in range(_SPF)], dtype=np.float32
    )


def _stream_all(flat: list, chunk: int, ctx: int) -> np.ndarray:
    """Reproduce OrpheusVoice.stream's chunk boundaries over a complete code list."""
    total = len(flat) // _CODES
    emitted, spf, out = 0, None, []
    while total - emitted >= chunk:
        audio, spf = decode_window(flat, emitted, emitted + chunk, ctx, _fake_decode, spf)
        out.append(audio)
        emitted += chunk
    if total > emitted:
        audio, spf = decode_window(flat, emitted, total, ctx, _fake_decode, spf)
        out.append(audio)
    return np.concatenate(out)


def _expected(n_frames: int) -> np.ndarray:
    return np.array([float(f) for f in range(n_frames) for _ in range(_SPF)], dtype=np.float32)


def test_streamed_chunks_reconstruct_audio_without_repeats():
    # 25 frames, 8-frame chunks, 4-frame context -> multiple seams, uneven final chunk.
    flat = _flat(25)
    out = _stream_all(flat, chunk=8, ctx=4)
    np.testing.assert_array_equal(out, _expected(25))


def test_no_context_is_still_exact():
    flat = _flat(20)
    out = _stream_all(flat, chunk=5, ctx=0)
    np.testing.assert_array_equal(out, _expected(20))


def test_context_larger_than_available_frames():
    # First chunk has fewer frames than the context window -> must not over-trim.
    flat = _flat(3)
    out = _stream_all(flat, chunk=10, ctx=8)  # one final chunk of 3 frames
    np.testing.assert_array_equal(out, _expected(3))


def test_single_chunk_matches_one_shot_decode():
    flat = _flat(6)
    out = _stream_all(flat, chunk=100, ctx=4)  # everything in one final window
    np.testing.assert_array_equal(out, _fake_decode(flat))


def test_seam_does_not_duplicate_the_context_region():
    # If the trim were wrong (mlx-audio's bug), the context frames would reappear and
    # the output would be longer than the true audio. Length must be exactly n*SPF.
    flat = _flat(17)
    out = _stream_all(flat, chunk=6, ctx=4)
    assert out.shape[0] == 17 * _SPF
