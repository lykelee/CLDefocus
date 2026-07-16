import os
from enum import Enum
from pathlib import Path

import numpy as np

from datasyn.synthesis.layered.scene import *
from datasyn.utils.lmdb.npystore import LmdbNpyStore


class LayerPatchStorage:
    """
    thread-safe
    """


class IOMode(Enum):
    READ = "READ"
    OVERWRITE = "OVERWRITE"


class LayerPatchStorage_LMDB(LayerPatchStorage):
    def __init__(
        self,
        file: os.PathLike,
        mode: IOMode = IOMode.READ,
    ):
        self._iomode = mode
        readonly = self.is_readonly

        file = Path(file)

        import os as _os

        from numcodecs import Blosc

        _spec = _os.environ.get("SYN_S3_CLST_CODEC", "zstd1").lower()
        _codec_map = {
            "zstd5": Blosc(cname="zstd", shuffle=1, clevel=5),
            "zstd1": Blosc(cname="zstd", shuffle=1, clevel=1),
            "lz4": Blosc(cname="lz4", shuffle=1, clevel=5),
            "lz4-1": Blosc(cname="lz4", shuffle=1, clevel=1),
            "none": None,
        }
        if _spec not in _codec_map:
            raise ValueError(f"Unknown SYN_S3_CLST_CODEC={_spec!r}")
        codec = _codec_map[_spec]

        # Layer data is regenerable, resumable pipeline output: disable
        # per-commit fsync (sync/metasync=False) so each write's small txns
        # don't each pay a disk flush. DB stays consistent on a process crash;
        # only a system power loss can drop the most recent txns, which a rerun
        # simply re-materializes. (No effect when readonly.)
        self._clst = LmdbNpyStore(
            file / "clst.lmdb",
            map_size=64 << 30,
            codec=codec,
            readonly=readonly,
            sync=False,
            metasync=False,
        )
        self._l2d = LmdbNpyStore(
            file / "l2d.lmdb",
            map_size=256 << 20,
            codec=None,
            readonly=readonly,
            sync=False,
            metasync=False,
        )
        self._photos = LmdbNpyStore(
            file / "photos.lmdb",
            map_size=256 << 20,
            codec=None,
            readonly=readonly,
            sync=False,
            metasync=False,
        )
        self._ofss = LmdbNpyStore(
            file / "offsets.lmdb",
            map_size=256 << 20,
            codec=None,
            readonly=readonly,
            sync=False,
            metasync=False,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    @property
    def is_readonly(self):
        return self._iomode == IOMode.READ

    @property
    def allow_overwrite(self):
        return self._iomode == IOMode.OVERWRITE

    def assert_writable(self):
        if self.is_readonly:
            raise Exception("This is opened as read-only!")

    def load_clst_idx(self, name: str):
        """Per-pixel integer level-index map (uint16, H×W) for online rebuild."""
        return np.asarray(self._clst.get(name))

    def load_depths(self, name: str, device: JaxDevice | None = None):
        l2d = self._l2d.get(name)
        l2d = jnp.asarray(l2d, device=device)
        return l2d

    def add_depth_layers(
        self,
        name: str,
        idx_map: np.ndarray,
        l2d: np.ndarray,
        photo: str,
        ofs: Tuple[int, int] = (0, 0),
    ):
        """
        Store the cluster index map + per-layer depths (no image layers).

        `idx_map`: per-pixel integer level index (`np.unique(clst_dpt, return_inverse=True)` reshaped to H x W).
        `l2d`: per-layer real depths (the `levels` from the same unique call).
        """
        self.assert_writable()
        self._clst.put(name, np.ascontiguousarray(idx_map, dtype=np.uint16))
        self._l2d.put(name, np.asarray(l2d))
        self._photos.put(name, np.array(photo, dtype="U32"))
        self._ofss.put(name, np.asarray(ofs))

    def remove_scene(self, name: str):
        self.assert_writable()
        self._clst.delete(name)
        self._l2d.delete(name)
        self._photos.delete(name)
        self._ofss.delete(name)

    def get_scene_names(self):
        names = [x for x in self._photos.keys()]
        return names

    def get_scene_photos(self):
        photos = [str(x) for x in self._photos.values()]
        return photos

    def get_scene_array_info(self, name: str):
        return self._clst.get_info(name)
