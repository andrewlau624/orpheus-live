"""Shared fakes for testing without real audio, models, or LLMs."""


class FakeClock:
    """Advanceable fake time source: t = clock.now(); clock.advance(1.5)."""

    def __init__(self, start: float = 0.0):
        self._t = start

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


class FakeVad:
    """Replays a scripted sequence of is_speech() results, then repeats the last."""

    def __init__(self, script: list[bool]):
        self._script = script
        self._i = 0

    def is_speech(self, frame: bytes, threshold: float | None = None) -> bool:
        return self.probability(frame, threshold) >= (threshold if threshold is not None else 0.5)

    def probability(self, frame: bytes, threshold: float | None = None) -> float:
        val = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return 1.0 if val else 0.0
