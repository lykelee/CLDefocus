"""
PsfStore: stores per-patch, per-layer Zernike PSFs.

Thin wrapper around BespokePSFStorage that uses the local IOMode.
All imports are deferred so this module is safe to import from spawn workers
(BespokePSFStorage itself imports JAX at module level).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp

from datasyn.synthesis.common import IOMode
from datasyn.synthesis.layered.storage import IOMode as _BespokeIOMode


class PSFMetadata(NamedTuple):
    lens: str
    sgncoc: list[float]

    def to_json(self):
        import json
        return json.dumps({"lens": self.lens, "sgncoc": [float(x) for x in self.sgncoc]})

    @classmethod
    def from_json(cls, j: str):
        import json
        try:
            d = json.loads(j)
            return PSFMetadata(lens=d["lens"], sgncoc=[float(x) for x in d["sgncoc"]])
        except Exception as e:
            raise ValueError("Wrong JSON") from e


class BespokePSFStorage:
    def __init__(self, path: os.PathLike, mode: _BespokeIOMode = _BespokeIOMode.READ):
        from numcodecs import Blosc

        from datasyn.utils.lmdb.npystore import LmdbNpyStore
        from datasyn.utils.sqlite.kvstore import KVStore

        self._iomode = mode
        path = Path(path)
        readonly = self._iomode == _BespokeIOMode.READ
        codec = Blosc(cname="zstd", clevel=3, shuffle=2)
        self._psfs = LmdbNpyStore(path / "psfs.lmdb", codec=codec, readonly=readonly)
        self._metadata_store = KVStore(path / "metadata.db", readonly=readonly)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    @property
    def is_readonly(self):
        return self._iomode == _BespokeIOMode.READ

    @property
    def allow_overwrite(self):
        return self._iomode == _BespokeIOMode.OVERWRITE

    def assert_writable(self):
        if self.is_readonly:
            raise Exception("This is opened as read-only!")

    def load_psfs(self, name: str, device=None):
        psfs = self._psfs.get(name)
        return jnp.asarray(psfs, device=device)

    def get_metadata(self, name: str) -> PSFMetadata:
        return PSFMetadata.from_json(self._metadata_store.get(name))

    def add_psfs(self, name: str, psfs, meta: PSFMetadata):
        self.assert_writable()
        self._psfs.put(name, psfs)
        self._metadata_store.set(name, meta.to_json())

    def delete_psfs(self, name: str):
        self.assert_writable()
        self._psfs.delete(name)
        self._metadata_store.delete(name)


class PsfStore:
    """IOMode.READ — read-only. IOMode.WRITE — read/write."""

    def __init__(self, path: Path, mode: IOMode = IOMode.READ) -> None:
        path = Path(path)
        if mode == IOMode.WRITE:
            path.mkdir(parents=True, exist_ok=True)
            lmdb_mode = _BespokeIOMode.OVERWRITE
        else:
            lmdb_mode = _BespokeIOMode.READ

        self._storage = BespokePSFStorage(path, mode=lmdb_mode)

    def add_psfs(self, name: str, psfs, meta) -> None:
        self._storage.add_psfs(name, psfs, meta)

    def load_psfs(self, name: str, device=None):
        return self._storage.load_psfs(name, device=device)

    def get_metadata(self, name: str):
        return self._storage.get_metadata(name)
