"""
Shared types for datasyn.synthesis.
"""

from enum import Enum
from typing import NamedTuple


class IOMode(Enum):
    READ = "READ"
    WRITE = "WRITE"


class CameraParam(NamedTuple):
    lens: str
    focusing: float
    sgncoc_max: float
