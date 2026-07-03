"""Flight recorder: session JSONL log + optional terminal traces.

Set `ORPHEUS_LIVE_DEBUG=1` to enable. Every event goes to a session log file
(`logs/YYYY-MM-DD_HH-MM-SS_PID.jsonl`) — one JSON line per event with thread name
and monotonic clock, so parallel activity is reconstructable. The same lines also
print to the terminal (dimmed) when debug is on.

`emit()` is always available to callers; it's a no-op when debug is off. This means
every source (capture, playback, conversation, speculation, remote TTS) can log
without conditional guards.
"""

import json
import os
import queue
import threading
import time
from pathlib import Path


def _ts() -> str:
    """Wall-clock timestamp for the log filename."""
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def _thread_name() -> str:
    try:
        return threading.current_thread().name
    except Exception:
        return "?"


class FlightRecorder:
    """Thread-safe pipeline logger; no-op unless `enabled` is set at startup."""

    def __init__(self):
        self.enabled = False
        self._t0 = time.monotonic()
        self._marks: dict[str, float] = {}
        self._lock = threading.Lock()
        self._q: queue.Queue[dict | None] = queue.Queue(maxsize=4096)
        self._log_path: Path | None = None
        self._printer: threading.Thread | None = None

    def enable(self, log_dir: str | Path = "logs") -> None:
        """Turn on tracing and open the session log file."""
        self.enabled = True
        pid = os.getpid()
        self._log_path = Path(log_dir) / f"{_ts()}_{pid}.jsonl"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        # Write header as the first line.
        with open(self._log_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "t": 0.0,
                        "thread": "main",
                        "event": "_session.start",
                        "pid": pid,
                        "log": str(self._log_path),
                    },
                    default=str,
                )
                + "\n"
            )
        self.emit("trace.on", log=str(self._log_path))
        if self._printer is None:
            self._printer = threading.Thread(target=self._print_loop, daemon=True)
            self._printer.name = "tracer"
            self._printer.start()

    def _print_loop(self) -> None:
        """Drain the queue: write every event to JSONL; echo curated ones to terminal."""
        # Import here to avoid circular import at module level.
        from .console import DIM, log

        while True:
            obj = self._q.get()
            if obj is None:
                break
            echo = obj.pop("_echo", True)
            # Write to log file (always, full fidelity).
            try:
                with open(self._log_path, "a") as f:
                    f.write(json.dumps(obj, default=str) + "\n")
            except OSError:
                pass
            if not echo:
                continue
            now = obj["t"]
            event = obj["event"]
            fields = {k: v for k, v in obj.items() if k not in ("t", "thread", "event")}
            parts = [f"⏱ +{now:8.3f} {event}"]
            for k, v in fields.items():
                if v is None:
                    continue  # logged to file for fidelity; omitted from the terminal line
                parts.append(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}")
            log("  " + " ".join(parts), DIM)

    def mark(self, name: str) -> None:
        """Anchor a moment for later `since()` deltas."""
        if not self.enabled:
            return
        with self._lock:
            self._marks[name] = time.monotonic()

    def since(self, name: str) -> float | None:
        """Seconds since `mark(name)`, or None if never marked."""
        if not self.enabled:
            return None
        with self._lock:
            t = self._marks.get(name)
        return None if t is None else time.monotonic() - t

    def emit(self, event: str, _echo: bool = True, **fields) -> None:
        """Queue one event; cheap and safe from any thread (incl. audio callback).

        Every call produces a JSON line with clock, thread name, and the event fields.
        `_echo=False` events go to the log file only (high-volume or already-visible
        content, e.g. per-chunk writes and console mirrors). No-op when debug is off.
        """
        if not self.enabled:
            return
        obj = {
            "t": round(time.monotonic() - self._t0, 5),
            "thread": _thread_name(),
            "event": event,
            "_echo": _echo,
        }
        obj.update(fields)
        try:
            self._q.put_nowait(obj)
        except queue.Full:
            pass  # drop on overflow; caller should not block


tracer = FlightRecorder()
