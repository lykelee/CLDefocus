"""
mydebugtoolkit — Low-friction debugging utilities for Python.

Quick start (recommended alias):
    import datasyn.utils.debug as d

Global singletons (zero-setup):
    d.hub       — share variables across stack frames
    d.box       — shortcut for d.hub.box  (primary read/write interface)
    d.flow      — conditional triggers (hit/every/once/prob)
    d.stopwatch — manual tic/toc timing

Shortcuts forwarded from the global stopwatch:
    d.tic(name)
    d.toc(name)
    d.measure(name)  — context manager

Inspection helpers (standalone functions):
    d.see(obj)        — dump all attributes
    d.structure(obj)  — show data shape without full contents

Underlying classes (for isolated / power-user instances):
    d.Hub, d.Flow, d.Stopwatch
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from datasyn.utils.debug._flow import Flow
from datasyn.utils.debug._hub import Hub
from datasyn.utils.debug._stopwatch import Stopwatch
from datasyn.utils.debug._xray import see, structure

# ---------------------------------------------------------------------------
# Global singletons
# ---------------------------------------------------------------------------

hub = Hub()
flow = Flow()
stopwatch = Stopwatch()

# Shortcut: d.box.x  instead of  d.hub.box.x
box = hub.box

# ---------------------------------------------------------------------------
# Stopwatch shortcuts
# ---------------------------------------------------------------------------


def tic(name: str = "default") -> None:
    """Start (or restart) the global stopwatch for name."""
    stopwatch.tic(name)


def toc(name: str = "default") -> float:
    """Stop the global stopwatch for name and print elapsed time."""
    return stopwatch.toc(name)


@contextmanager
def measure(name: str = "default") -> Iterator[None]:
    """Context manager: tic on enter, toc on exit (global stopwatch)."""
    with stopwatch.measure(name):
        yield


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # singletons
    "hub",
    "box",
    "flow",
    "stopwatch",
    # stopwatch shortcuts
    "tic",
    "toc",
    "measure",
    # x-ray
    "see",
    "structure",
    # classes
    "Hub",
    "Flow",
    "Stopwatch",
]
