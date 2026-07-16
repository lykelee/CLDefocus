"""
PatchStore: stores linear patches, depth patches, and patch definitions.

Layout on disk::

    <store_dir>/
        linears/    <- LmdbNpyStore   key=patch_name, value=uint16 HxWx3
        depths/     <- LmdbNpyStore   key=patch_name, value=float32 HxW
        definitions.db  <- SQLiteTableStore  primary_key="name"
"""

from __future__ import annotations

from pathlib import Path

from datasyn.synthesis.common import IOMode
from datasyn.synthesis.patch_define import PatchDefine
from datasyn.utils.lmdb.npystore import LmdbNpyStore
from datasyn.utils.sqlite.tablestore import SQLiteTableStore

_DEF_COLUMNS: dict[str, type] = {
    "name": str,
    "photo": str,
    "photo_size_h": int,
    "photo_size_w": int,
    "patch_offset_i": float,
    "patch_offset_j": float,
    "patch_size_h": int,
    "patch_size_w": int,
    "field_x": float,
    "field_y": float,
    "aug_theta": float,
    "aug_scale": float,
    "aug_flip": int,
    "aug_exposure": float,
    "depth_min": float,
    "depth_max": float,
}


class PatchStore:
    """
    Thread-safe persistent store for patch data.

    IOMode.READ  — read-only; directory must already exist.
    IOMode.WRITE — read/write; directory is created if absent.

    ``add()`` is idempotent: safe to call again if a previous run was
    interrupted before progress was recorded.
    """

    def __init__(self, path: Path, mode: IOMode = IOMode.READ) -> None:
        path = Path(path)
        readonly = mode == IOMode.READ

        if not readonly:
            path.mkdir(parents=True, exist_ok=True)

        from numcodecs import Blosc

        codec = Blosc(cname="zstd", shuffle=1, clevel=5)

        self._lins = LmdbNpyStore(
            path / "linears",
            codec=codec,
            readonly=readonly,
        )
        self._dpts = LmdbNpyStore(
            path / "depths",
            codec=codec,
            readonly=readonly,
        )
        self._table = SQLiteTableStore(
            path / "definitions.db",
            _DEF_COLUMNS,
            primary_key="name",
            readonly=readonly,
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @staticmethod
    def _def_row(patdef: PatchDefine) -> dict:
        return {
            "name": patdef.name,
            "photo": patdef.photo,
            "photo_size_h": int(patdef.photo_size[0]),
            "photo_size_w": int(patdef.photo_size[1]),
            "patch_offset_i": float(patdef.patch_offset[0]),
            "patch_offset_j": float(patdef.patch_offset[1]),
            "patch_size_h": int(patdef.patch_size[0]),
            "patch_size_w": int(patdef.patch_size[1]),
            "field_x": float(patdef.f_xy[0]),
            "field_y": float(patdef.f_xy[1]),
            "aug_theta": float(patdef.aug.theta),
            "aug_scale": float(patdef.aug.scale),
            "aug_flip": int(patdef.aug.flip),
            "aug_exposure": float(patdef.aug.exposure),
            "depth_min": float(patdef.depth_min),
            "depth_max": float(patdef.depth_max),
        }

    def add(self, patdef: PatchDefine, lin, dpt) -> None:
        """
        Store a patch.  LMDB writes are idempotent; SQLite insert is skipped
        if the name already exists (handles retries after crash).
        """
        self._lins.put(patdef.name, lin)
        self._dpts.put(patdef.name, dpt)
        if not self._table.get("name", patdef.name):
            self._table.insert(self._def_row(patdef))

    def add_many(self, items) -> None:
        """
        Batch-store many patches in few transactions (one LMDB txn per
        array store + one SQLite insert_many), instead of per-patch commits.

        ``items``: iterable of ``(patdef, lin, dpt)``. Idempotent: names already
        present in the definitions table are skipped for the SQLite insert.
        """
        items = list(items)
        if not items:
            return
        self._lins.put_many([(p.name, lin) for p, lin, _ in items])
        self._dpts.put_many([(p.name, dpt) for p, _, dpt in items])
        existing = set(self._table.get_column("name"))
        rows = [self._def_row(p) for p, _, _ in items if p.name not in existing]
        if rows:
            self._table.insert_many(rows)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_all_names(self) -> list[str]:
        return sorted(self._table.get_column("name"))

    def load_patdef(self, name: str) -> PatchDefine:
        rows = self._table.get("name", name)
        if not rows:
            raise KeyError(f"PatchStore: unknown patch '{name}'")
        return PatchDefine.from_dict(rows[0])

    def load_linear(self, name: str):
        return self._lins.get(name)

    def load_depth(self, name: str):
        return self._dpts.get(name)

    def load_photo_map(self) -> dict[str, str]:
        """Bulk-read {patch_name: photo_name} for all patches in one query."""
        return {r["name"]: r["photo"] for r in self._table.get_all()}
