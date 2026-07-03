"""Tests for AudioSink's jitter buffer: arming, rebuffer-on-underrun, lead-aware pacing.

The real sink owns an sd.OutputStream whose callback pulls from the buffer; here the
stream is stubbed out, the callback is driven by hand, and the clock is faked, so the
arming state machine can be tested sample-by-sample without a sound device.
"""

import threading

import numpy as np
import pytest

from orpheus_live.audio import playback
from orpheus_live.config import Settings


class _FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = _FakeClock()
    monkeypatch.setattr(playback.time, "monotonic", c)
    return c


@pytest.fixture
def sink(monkeypatch, clock):
    monkeypatch.setattr(playback.sd, "OutputStream", _FakeStream)
    settings = Settings(
        tts_sample_rate=1000,  # round numbers: 1 sample = 1ms
        tts_prebuffer_s=0.1,  # arm at 100 samples
        tts_rebuffer_s=0.4,  # post-underrun arm floor: 400 samples
    )
    s = playback.AudioSink(settings)
    yield s
    s.close()


def _pull(sink, frames):
    """Drive the stream callback once; return the samples it emitted."""
    out = np.empty((frames, 1), dtype=np.float32)
    sink._callback(out, frames, None, None)
    return out.reshape(-1)


def _chunk(n):
    return np.full(n, 0.5, dtype=np.float32)


# -- plain (unpaced) arming: prebuffer floor and rebuffer-on-underrun ----------


def test_silent_until_prebuffer_met(sink):
    sink.begin(0)
    sink.write(_chunk(50), 0)
    assert not sink._armed
    assert np.all(_pull(sink, 64) == 0.0)  # not armed -> silence, buffer untouched
    sink.write(_chunk(60), 0)  # 110 buffered >= 100
    assert sink._armed
    assert np.any(_pull(sink, 64) != 0.0)


def test_underrun_disarms_and_raises_floor_to_rebuffer(sink):
    sink.begin(0)
    sink.write(_chunk(120), 0)
    assert sink._armed
    _pull(sink, 120)
    _pull(sink, 64)  # buffer dry mid-stream -> underrun
    assert not sink._armed
    assert sink._had_underrun
    # Refills below the raised floor stay silent...
    sink.write(_chunk(200), 0)
    assert not sink._armed
    assert np.all(_pull(sink, 64) == 0.0)
    # ...and playback resumes only once the bigger cushion is rebuilt.
    sink.write(_chunk(250), 0)
    assert sink._armed
    assert np.any(_pull(sink, 64) != 0.0)


def test_raised_floor_is_sticky_across_utterances(sink):
    sink.begin(0)
    sink.write(_chunk(120), 0)
    _pull(sink, 120)
    _pull(sink, 64)  # underrun -> floor raised
    sink.begin(1)
    sink.write(_chunk(120), 1)  # meets prebuffer, not rebuffer
    assert not sink._armed  # a link that stalled once will stall again


def test_flush_drains_then_disarms_without_flagging_underrun(sink):
    sink.begin(0)
    sink.write(_chunk(120), 0)
    # flush() blocks until the callback empties the buffer, so drive it from a thread.
    t = threading.Thread(target=sink.flush, args=(0,))
    t.start()
    while sink._buffered > 0:
        _pull(sink, 64)
    t.join(timeout=2.0)
    assert not t.is_alive()
    # Post-drain callbacks (before the next begin) are silence, not "underruns".
    assert not sink._armed
    assert np.all(_pull(sink, 64) == 0.0)
    assert not sink._had_underrun


def test_clear_fades_out_then_goes_silent_and_drops_stale_writes(sink):
    sink.begin(0)
    sink.write(_chunk(200), 0)  # armed (>= prebuffer)
    sink.clear()
    # Interrupt fade: the buffer is kept and ramped out, not dropped instantly.
    assert sink._fade_left > 0
    buffered_mid_fade = sink._buffered
    sink.write(_chunk(200), 0)  # stale epoch -> dropped even mid-fade
    assert sink._buffered == buffered_mid_fade  # stale write ignored
    # Drive the callback past the fade window; output ends at zero and buffer empties.
    total = 0
    for _ in range(20):
        out = _pull(sink, 64)
        total += out.shape[0]
        if sink._fade_left == 0:
            break
    assert sink._fade_left == 0
    assert sink._buffered == 0
    assert not sink._armed
    assert np.all(_pull(sink, 64) == 0.0)  # silent afterwards


def test_clear_without_audible_audio_drops_clean(sink):
    sink.begin(0)
    sink.write(_chunk(50), 0)  # below prebuffer -> not armed, nothing audible
    sink.clear()
    assert sink._fade_left == 0  # nothing to fade
    assert sink._buffered == 0
    assert np.all(_pull(sink, 64) == 0.0)


# -- lead-aware pacing: hold back exactly the deficit of a sub-realtime source --


def test_paced_fast_source_arms_at_prebuffer(sink, clock):
    sink.begin(0)
    sink.pace(4.0)  # expect ~4000 samples
    sink.write(_chunk(300), 0)  # rate unknown on the first write -> hold
    assert not sink._armed
    clock.now += 0.1
    sink.write(_chunk(300), 0)  # 600 samples in 0.1s -> 6x realtime -> no lead needed
    assert sink._armed


def test_paced_slow_source_holds_until_lead_covers_deficit(sink, clock):
    sink.begin(0)
    sink.pace(2.0)  # expect 2000 samples total
    # Source delivers 250 samples per 500ms -> r = 0.5x realtime. Gap-free playback
    # needs (1-r)*T = 1000 samples of lead before starting.
    armed_at = None
    for _ in range(8):
        sink.write(_chunk(250), 0)
        if sink._armed and armed_at is None:
            armed_at = sink._buffered
        clock.now += 0.5
    assert armed_at is not None, "never armed"
    assert armed_at >= 1000  # held at least the theoretical minimum lead
    assert armed_at < 2000  # ...but did NOT wait for the whole sentence


def test_paced_slow_source_plays_through_without_underrun(sink, clock):
    """End-to-end: 0.5x source, playback starts when armed, buffer never runs dry."""
    sink.begin(0)
    sink.pace(2.0)
    played = 0
    for _ in range(8):  # 8 writes x 250 samples = 2000 total, one write per 500ms
        sink.write(_chunk(250), 0)
        clock.now += 0.5
        if sink._armed:
            before = sink._buffered
            _pull(sink, 500)  # play the 500ms that just elapsed
            drawn = before - sink._buffered
            assert drawn == 500, "underrun: buffer ran dry mid-sentence"
            played += drawn
    # Drain the rest.
    with sink._cv:
        sink._draining = True
    while sink._buffered > 0:
        before = sink._buffered
        _pull(sink, 500)
        played += before - sink._buffered
    assert played == 2000  # every sample accounted for, none dropped
    assert not sink._had_underrun


def test_long_sentence_hold_is_capped_not_the_full_lead(sink, clock):
    # A long sentence on a slow source: the ideal lead is several seconds, but the cap
    # (tts_max_hold_s) bounds it so playback starts fast instead of sitting in silence.
    sink.begin(0)
    sink.pace(8.0)  # 8000 samples expected -> ideal lead would be multiple seconds
    sink.write(_chunk(400), 0)
    clock.now += 0.5
    sink.write(_chunk(200), 0)  # rate ~0.4x -> ideal (1-r)*8000 is well over the cap
    assert sink._arm_target(clock.now) == sink._max_hold  # capped, not the full lead


def test_pace_reseats_the_segment_per_sentence(sink, clock):
    # Each sentence is its own paced segment: pace() re-seats the rate measurement and
    # widens the backpressure cap so a full sentence can be buffered while it's held.
    sink.begin(0)
    sink.pace(1.0)
    sink.write(_chunk(300), 0)
    clock.now += 0.3
    sink.write(_chunk(300), 0)
    sink.pace(1.0)  # next sentence -> fresh segment
    assert sink._seg_t0 is None  # measurement reset for the new sentence
    assert sink._seg_written == 0


def test_pace_widens_backpressure_cap_to_hold_a_full_sentence(sink):
    sink.begin(0)
    assert sink._max_buffered == sink._base_cap
    sink.pace(10.0)  # a long sentence must be bufferable while held
    assert sink._max_buffered >= 10 * sink._sr


def test_on_audible_fires_when_playback_arms_not_on_first_write(sink, clock):
    heard = []
    sink.begin(0, on_audible=lambda: heard.append(sink._buffered))
    sink.pace(2.0)
    sink.write(_chunk(250), 0)
    assert heard == []  # buffered but not audible yet
    clock.now += 0.5
    for _ in range(6):
        sink.write(_chunk(250), 0)
        clock.now += 0.5
    assert len(heard) == 1  # fired exactly once, at arming


def test_flush_fires_on_audible_for_short_utterances(sink):
    heard = []
    sink.begin(0, on_audible=lambda: heard.append(True))
    sink.write(_chunk(50), 0)  # below prebuffer -> never armed by write
    t = threading.Thread(target=sink.flush, args=(0,))
    t.start()
    while sink._buffered > 0:
        _pull(sink, 64)
    t.join(timeout=2.0)
    assert heard == [True]


def test_flush_does_not_fire_on_audible_when_nothing_written(sink):
    heard = []
    sink.begin(0, on_audible=lambda: heard.append(True))
    t = threading.Thread(target=sink.flush, args=(0,))
    t.start()
    t.join(timeout=2.0)
    assert heard == []


# -- session rate reuse: sentence N+1 arms on N's measured speed, not from scratch --


def test_second_sentence_arms_immediately_from_session_rate(sink, clock):
    sink.begin(0)
    # Sentence 1 at 2x realtime: first write can't prove a rate (hold), second can.
    sink.pace(2.0)
    sink.write(_chunk(500), 0)
    assert not sink._armed
    clock.now += 0.25
    sink.write(_chunk(500), 0)  # 500 samples in 0.25s -> 2x -> arm at the floor
    assert sink._armed
    _pull(sink, 1000)  # play sentence 1 fully
    # Sentence 2: pace() folds the ~2x rate into the session estimate; the seam dry-out
    # disarms silently, and the FIRST write re-arms — no second write needed to re-prove
    # the rate, so the boundary pause is just one chunk's arrival time.
    sink.pace(2.0)
    _pull(sink, 64)  # dry-out at the seam
    assert not sink._armed
    assert not sink._had_underrun
    sink.write(_chunk(250), 0)
    assert sink._armed


def test_seam_dryout_is_not_a_sticky_underrun(sink, clock):
    sink.begin(0)
    # A short segment fully delivered, then played out to empty.
    sink.pace(0.5)
    sink.write(_chunk(500), 0)
    clock.now += 0.1
    sink._armed = True  # simulate arming + drain by hand (flush would block)
    _pull(sink, 500)  # play it all
    # Next sentence is paced -> the tail running dry is a seam, not a stall.
    sink.pace(2.0)
    _pull(sink, 64)  # callback fires with empty buffer at the seam
    assert not sink._had_underrun


def test_idle_between_sentences_does_not_dilute_rate(sink, clock):
    sink.begin(0)
    sink.pace(2.0)
    # Deliver at ~2x realtime: 500 samples per 0.25s.
    sink.write(_chunk(500), 0)
    clock.now += 0.25
    sink.write(_chunk(500), 0)
    clock.now += 0.25
    sink.write(_chunk(500), 0)
    # Long idle (waiting on the LLM for sentence 2) before the seam closes the segment.
    clock.now += 5.0
    sink.pace(2.0)  # folds sentence 1's rate
    assert sink._rate_ema is not None
    assert sink._rate_ema > 1.5  # idle time excluded -> rate reflects delivery, ~2x


# -- estimator self-calibration: expected vs delivered corrects future holds ----


def test_estimate_calibrates_down_after_completed_overshoot(sink, clock):
    sink.begin(0)
    sink.pace(4.0)  # estimate 4000 samples
    # Deliver only 2000 (estimator overshot 2x), spread over time so the rate folds too.
    for _ in range(4):
        sink.write(_chunk(500), 0)
        clock.now += 0.5
    sink.pace(4.0)  # closes segment 1 -> calibrates; opens segment 2
    assert sink._est_scale < 1.0  # learned the overshoot
    assert sink._expected < 4000  # the same text estimate now holds less audio


def test_estimate_scale_is_clamped(sink, clock):
    sink.begin(0)
    sink.pace(10.0)  # wildly high estimate
    sink.write(_chunk(600), 0)
    clock.now += 0.5
    sink.write(_chunk(600), 0)
    sink.pace(1.0)
    assert sink._est_scale >= sink._EST_SCALE_MIN  # one outlier can't run away


def test_cancelled_segment_does_not_calibrate(sink, clock):
    sink.begin(0)
    sink.pace(4.0)
    sink.write(_chunk(600), 0)  # only a fragment delivered...
    clock.now += 0.5
    sink.clear()  # ...because the user barged in, not because the estimate was high
    sink.begin(1)
    assert sink._est_scale == 1.0


# -- rate pessimism: marginal sources hold a little instead of nothing ----------


def test_marginal_source_rate_still_holds_some_lead(sink, clock):
    sink.begin(0)
    sink.pace(4.0)
    sink.write(_chunk(500), 0)
    clock.now += 0.5
    sink.write(_chunk(525), 0)  # 525 samples in 0.5s -> measured 1.05x, barely realtime
    # 1.05 * 0.85 pessimism = 0.89 -> holds ~(1-0.89)*4000 + margin, not the bare floor.
    assert sink._arm_target(clock.now) > sink._prebuffer


def test_fast_source_still_arms_at_floor(sink, clock):
    sink.begin(0)
    sink.pace(4.0)
    sink.write(_chunk(500), 0)
    clock.now += 0.25
    sink.write(_chunk(500), 0)  # 2x realtime -> pessimism-adjusted 1.7, still >= 1
    assert sink._armed
