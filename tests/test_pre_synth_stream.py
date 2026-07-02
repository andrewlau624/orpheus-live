"""Tests for PreSynthStream: buffered speculative chunks drain in order, block, and cancel."""

import threading
import time

import numpy as np

from orpheus_live.audio.playback import PreSynthStream


def _chunk(v: int) -> np.ndarray:
    return np.full(4, float(v), dtype=np.float32)


def test_buffered_chunks_drain_in_order_after_finish():
    ps = PreSynthStream("hi")
    ps.add(_chunk(1))
    ps.add(_chunk(2))
    ps.finish()
    out = [int(c[0]) for c in ps.iter_chunks()]
    assert out == [1, 2]


def test_consumer_blocks_until_next_chunk_then_finish():
    ps = PreSynthStream("hi")
    got: list[int] = []

    def consume():
        got.extend(int(c[0]) for c in ps.iter_chunks())

    t = threading.Thread(target=consume)
    t.start()
    ps.add(_chunk(7))
    time.sleep(0.05)
    assert got == [7]  # first chunk consumed; consumer now blocks for more
    ps.add(_chunk(8))
    ps.finish()
    t.join(timeout=2)
    assert got == [7, 8]


def test_cancel_unblocks_consumer_and_stops_iteration():
    ps = PreSynthStream("hi")
    got: list[int] = []
    done = threading.Event()

    def consume():
        got.extend(int(c[0]) for c in ps.iter_chunks())
        done.set()

    t = threading.Thread(target=consume)
    t.start()
    ps.add(_chunk(1))
    time.sleep(0.05)
    ps.cancel()  # must release the blocked consumer even though not finished normally
    assert done.wait(timeout=2)
    assert got == [1]
    assert ps.cancelled


def test_cancelled_flag_lets_producer_stop_early():
    ps = PreSynthStream("hi")
    assert ps.cancelled is False
    ps.cancel()
    assert ps.cancelled is True
