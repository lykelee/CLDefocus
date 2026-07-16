"""
DefocusDataset: abstract interface for datasets providing RAW and depth files.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class DefocusDataset(ABC):
    """
    Abstract interface for a defocus synthesis dataset.

    A dataset is a collection of named scenes, each providing:
      - A RAW photo file
      - A depth map file
    """

    @abstractmethod
    def get_names(self) -> list[str]:
        """Return all scene names in a consistent order."""
        ...

    @abstractmethod
    def get_raw(self, name: str) -> Path:
        """Path to the RAW file for the given scene."""
        ...

    @abstractmethod
    def get_depth(self, name: str) -> Path:
        """Path to the depth map file for the given scene."""
        ...


class FilesystemDefocusDataset(DefocusDataset):
    """
    A DefocusDataset backed by two flat directories: one each for RAW and
    depth files.  Files are matched by stem name.

    Names are anchored to depth_dir — the set of stems found there defines
    the dataset.

    Parameters
    ----------
    raw_dir, depth_dir
        Directories containing the respective files.
    raw_suffix
        Extension of RAW files, e.g., ``".dng"``, ``".cr2"``.
    depth_suffix
        Extension of depth files, e.g., ``".npz"`` or ``".npy"``.
    """

    def __init__(
        self,
        raw_dir: Path,
        depth_dir: Path,
        raw_suffix: str = ".dng",
        depth_suffix: str = ".npz",
    ) -> None:
        self._raw_dir = Path(raw_dir)
        self._depth_dir = Path(depth_dir)
        self._raw_suffix = raw_suffix
        self._depth_suffix = depth_suffix

        # Names determined by depth_dir contents (depth is the anchor).
        self._names: list[str] = sorted(
            p.stem
            for p in self._depth_dir.iterdir()
            if p.is_file() and p.suffix.lower() == depth_suffix.lower()
        )

    def get_names(self) -> list[str]:
        return list(self._names)

    def get_raw(self, name: str) -> Path:
        return self._raw_dir / f"{name}{self._raw_suffix}"

    def get_depth(self, name: str) -> Path:
        return self._depth_dir / f"{name}{self._depth_suffix}"
