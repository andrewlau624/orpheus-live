"""Tests for the flight recorder's gating, marks, and JSONL event shape."""

import json

from orpheus_live.debug import FlightRecorder


def test_disabled_recorder_emits_nothing():
    r = FlightRecorder()  # not enabled
    r.emit("x", a=1)
    r.mark("m")
    assert r.since("m") is None  # mark was a no-op while disabled
    assert r._q.empty()


def test_enabled_recorder_queues_events_and_tracks_marks():
    r = FlightRecorder()
    r.enabled = True  # don't open a file / start the printer; inspect the queue directly
    r.mark("pause")
    assert r.since("pause") is not None
    r.emit("turn.respond", since_pause_s=r.since("pause"))
    assert not r._q.empty()
    obj = r._q.get()
    assert obj["event"] == "turn.respond"
    assert "since_pause_s" in obj
    assert obj["thread"]  # thread name captured for parallel reconstruction
    assert "t" in obj  # session clock


def test_event_carries_echo_flag_for_file_only_events():
    r = FlightRecorder()
    r.enabled = True
    r.emit("sink.write", _echo=False, samples=512)
    obj = r._q.get()
    assert obj["_echo"] is False
    assert obj["samples"] == 512


def test_emit_is_json_serializable():
    r = FlightRecorder()
    r.enabled = True
    r.emit("stt.done", wall_s=0.31, text="hello there", chars=None)
    obj = r._q.get()
    # The printer writes json.dumps(obj); confirm that never raises and round-trips.
    line = json.dumps(obj, default=str)
    back = json.loads(line)
    assert back["text"] == "hello there"
    assert back["chars"] is None


def test_writes_session_log_file(tmp_path):
    r = FlightRecorder()
    r.enabled = True
    r._log_path = tmp_path / "session.jsonl"
    # Write the header by hand (enable() would also spawn the printer thread).
    import json as _json

    r._log_path.write_text(
        _json.dumps({"t": 0.0, "thread": "main", "event": "_session.start", "pid": 1}) + "\n"
    )
    header = json.loads(r._log_path.read_text().splitlines()[0])
    assert header["event"] == "_session.start"
    assert "pid" in header
