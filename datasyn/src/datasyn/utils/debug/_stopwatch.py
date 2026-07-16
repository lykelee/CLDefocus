from __future__ import annotations

import time as _time
from contextlib import contextmanager
from typing import Iterator


class Stopwatch:
    """Dead-simple manual stopwatch for timing code sections."""

    def __init__(self) -> None:
        self._starts: dict[str, float] = {}
        self._laps: dict[str, list[float]] = {}

    def tic(self, name: str = "default") -> None:
        """Start (or restart) the timer for name."""
        self._starts[name] = _time.perf_counter()

    def toc(self, name: str = "default") -> float:
        """Stop timer for name, print elapsed time, and return seconds."""
        if name not in self._starts:
            print(f"[stopwatch] {name!r}: no tic() called")
            return 0.0
        elapsed = _time.perf_counter() - self._starts[name]
        self._laps.setdefault(name, []).append(elapsed)
        print(f"[stopwatch] {name!r}: {_fmt(elapsed)}")
        return elapsed

    @contextmanager
    def measure(self, name: str = "default") -> Iterator[None]:
        """Context manager: tic on enter, toc on exit."""
        self.tic(name)
        try:
            yield
        finally:
            self.toc(name)

    def stats(self, name: str = "default") -> None:
        """Print summary stats over all recorded laps for name."""
        laps = self._laps.get(name)
        if not laps:
            print(f"[stopwatch] {name!r}: no laps recorded")
            return
        n = len(laps)
        avg = sum(laps) / n
        print(
            f"[stopwatch] {name!r}: n={n}"
            f"  avg={_fmt(avg)}  min={_fmt(min(laps))}  max={_fmt(max(laps))}"
        )

    def reset(self, name: str | None = None) -> None:
        """Reset timer(s). Pass a name to reset one, or None to reset all."""
        if name is None:
            self._starts.clear()
            self._laps.clear()
        else:
            self._starts.pop(name, None)
            self._laps.pop(name, None)

    def __repr__(self) -> str:
        return f"<Stopwatch active={list(self._starts.keys())}>"


def _fmt(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:.3f}s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.3f}ms"
    return f"{seconds * 1e6:.3f}µs"
