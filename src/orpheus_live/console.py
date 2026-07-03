"""Console colors and logging helper."""

import threading

DIM = "\033[2m"
YOU = "\033[96m"
AI = "\033[92m"
SYS = "\033[93m"
RESET = "\033[0m"


def log(msg: str, color: str = SYS) -> None:
    print(f"{color}{msg}{RESET}", flush=True)
    # Mirror every console line into the session flight recorder, so the log file
    # holds the full transcript (Voice:/You:/thoughts) interleaved with trace events.
    # The recorder's own printer thread ("tracer") is skipped to avoid recursion.
    from .debug import tracer

    if tracer.enabled and threading.current_thread().name != "tracer":
        tracer.emit("console", _echo=False, text=msg)
