"""
LayerStore: stores layered depth scenes (LMDB-backed).

Thin wrapper around LayerPatchStorage_LMDB that uses the local IOMode.
"""

from __future__ import annotations

from pathlib import Path

from datasyn.synthesis.common import IOMode


class LayerStore:
    """
    IOMode.READ  — read-only.
    IOMode.WRITE — read/write; directory is created if absent.
    """

    def __init__(self, path: Path, mode: IOMode = IOMode.READ) -> None:
        from datasyn.synthesis.layered.storage import (
            IOMode as _LIOMode,
        )
        from datasyn.synthesis.layered.storage import (
            LayerPatchStorage_LMDB,
        )

        path = Path(path)
        if mode == IOMode.WRITE:
            path.mkdir(parents=True, exist_ok=True)
            lmdb_mode = _LIOMode.OVERWRITE
        else:
            lmdb_mode = _LIOMode.READ

        self._storage = LayerPatchStorage_LMDB(path, mode=lmdb_mode)

    def add_depth_layers(
        self,
        name: str,
        idx_map,
        l2d,
        photo: str,
        ofs: tuple[int, int] = (0, 0),
    ) -> None:
        self._storage.add_depth_layers(name, idx_map, l2d, photo, ofs)

    def load_clst_idx(self, name: str):
        """Per-pixel integer level-index map (H×W uint16) for online rebuild."""
        return self._storage.load_clst_idx(name)

    def load_depths(self, name: str, device=None):
        return self._storage.load_depths(name, device=device)

    def get_all_names(self) -> list[str]:
        return sorted(self._storage.get_scene_names())
