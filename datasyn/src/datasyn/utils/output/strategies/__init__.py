"""Output management strategies."""

from .overwrite import OverwriteStrategy
from .rotating import RotatingStrategy

__all__ = [
    "OverwriteStrategy",
    "RotatingStrategy",
]
