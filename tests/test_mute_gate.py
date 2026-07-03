"""Tests for MuteGate: lag-aware mute with a force-unmute watchdog."""

import time

from orpheus_live.audio.capture import MuteGate


def test_set_and_clear_toggle_state():
    g = MuteGate(max_mute_s=0.0)  # 0 -> no watchdog
    assert not g.is_set()
    g.set()
    assert g.is_set()
    g.clear()
    assert not g.is_set()


def test_watchdog_force_unmutes_after_cap():
    g = MuteGate(max_mute_s=0.05)
    g.set()
    assert g.is_set()
    time.sleep(0.12)
    assert not g.is_set()  # watchdog fired


def test_clear_before_cap_cancels_watchdog():
    g = MuteGate(max_mute_s=0.05)
    g.set()
    g.clear()  # audible arrived first
    assert not g.is_set()
    time.sleep(0.12)
    assert not g.is_set()  # stayed clear; no spurious flip


def test_new_mute_is_not_unmuted_by_stale_watchdog():
    g = MuteGate(max_mute_s=0.05)
    g.set()  # gen 1
    time.sleep(0.03)
    g.clear()
    g.set()  # gen 2, fresh cap
    time.sleep(0.04)  # gen-1 timer would fire here; must be a no-op
    assert g.is_set()  # gen 2 still owns the mute
    time.sleep(0.05)
    assert not g.is_set()  # gen 2's own watchdog eventually fires


def test_zero_cap_never_auto_unmutes():
    g = MuteGate(max_mute_s=0.0)
    g.set()
    time.sleep(0.1)
    assert g.is_set()  # no watchdog when disabled
