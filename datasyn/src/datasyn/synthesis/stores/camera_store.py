"""
CameraStore: stores per-patch camera parameters (lens choice + focusing).

Layout on disk::

    <store_dir>   <- single SQLite file (pass the .db path)
"""

from __future__ import annotations

from pathlib import Path

from datasyn.synthesis.common import CameraParam, IOMode
from datasyn.utils.sqlite.tablestore import SQLiteTableStore

_CAM_COLUMNS: dict[str, type] = {
    "name": str,
    "lens": str,
    "focusing": float,
    "sgncoc_max": float,
}


class CameraStore:
    """
    Thread-safe persistent store for camera parameter assignments.

    IOMode.READ  — read-only.
    IOMode.WRITE — read/write; parent directory is created if absent.

    ``add()`` is idempotent.
    """

    def __init__(self, path: Path, mode: IOMode = IOMode.READ) -> None:
        path = Path(path)
        readonly = mode == IOMode.READ

        if not readonly:
            path.parent.mkdir(parents=True, exist_ok=True)

        self._table = SQLiteTableStore(
            path,
            _CAM_COLUMNS,
            primary_key="name",
            readonly=readonly,
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, name: str, param: CameraParam) -> None:
        """Store a camera parameter.  Skips insert if name already exists."""
        if not self._table.get("name", name):
            self._table.insert(
                {
                    "name": name,
                    "lens": str(param.lens),
                    "focusing": float(param.focusing),
                    "sgncoc_max": float(param.sgncoc_max),
                }
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(self, name: str) -> CameraParam:
        rows = self._table.get("name", name)
        if not rows:
            raise KeyError(f"CameraStore: unknown patch '{name}'")
        r = rows[0]
        return CameraParam(
            lens=r["lens"], focusing=r["focusing"], sgncoc_max=r["sgncoc_max"]
        )

    def load_all(self) -> dict[str, CameraParam]:
        return {
            r["name"]: CameraParam(
                lens=r["lens"], focusing=r["focusing"], sgncoc_max=r["sgncoc_max"]
            )
            for r in self._table.get_all()
        }

    def get_all_names(self) -> list[str]:
        return sorted(self._table.get_column("name"))
