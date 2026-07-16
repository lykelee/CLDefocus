from __future__ import annotations

import random


class Flow:
    """State-aware counters and triggers for conditional debugging."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def _inc(self, name: str) -> int:
        self._counts[name] = self._counts.get(name, 0) + 1
        return self._counts[name]

    def hit(self, name: str, n: int) -> bool:
        """Return True only on the nth call for this name."""
        return self._inc(name) == n

    def every(self, name: str, n: int) -> bool:
        """Return True every nth call for this name."""
        return self._inc(name) % n == 0

    def once(self, name: str) -> bool:
        """Return True only the very first call for this name."""
        return self._inc(name) == 1

    def prob(self, p: float) -> bool:
        """Return True with probability p (0.0–1.0), independent of name."""
        return random.random() < p

    def count(self, name: str) -> int:
        """Return current call count for a name (without incrementing)."""
        return self._counts.get(name, 0)

    def reset(self, name: str | None = None) -> None:
        """Reset counter(s). Pass a name to reset one, or None to reset all."""
        if name is None:
            self._counts.clear()
        else:
            self._counts.pop(name, None)

    def __repr__(self) -> str:
        return f"<Flow counts={self._counts}>"
