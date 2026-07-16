"""Output management utilities.

This module provides a simple system for managing output directories across
multiple runs of a program, with different strategies for cleanup and retention.

Strategies:
    - OverwriteStrategy: Set-based cleanup, exploits overlap between runs
    - RotatingStrategy: Fresh directory per run with symlink switching

Example:
    >>> from pathlib import Path
    >>> from datasyn.utils.output import OutputManager, OverwriteStrategy
    >>>
    >>> strategy = OverwriteStrategy(Path("./outputs"))
    >>> manager = OutputManager(strategy)
    >>>
    >>> expected = {Path(f"chunk_{i:04d}.bin") for i in range(100)}
    >>> with manager.run(expected) as out_dir:
    ...     for i in range(100):
    ...         (out_dir / f"chunk_{i:04d}.bin").write_bytes(b"data")
"""

from .manager import OutputManager
from .protocol import OutputStrategy
from .strategies import OverwriteStrategy, RotatingStrategy

__all__ = [
    "OutputManager",
    "OutputStrategy",
    "OverwriteStrategy",
    "RotatingStrategy",
]
