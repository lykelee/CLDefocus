from __future__ import annotations

import io
import json
import os
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Tuple

import lmdb
import numpy as np
import numpy.lib.format as npfmt
from numpy.typing import NDArray

try:
    from numcodecs.abc import Codec as NumcodecsCodec
    from numcodecs.registry import get_codec as numcodecs_get_codec
except ImportError:  # optional dependency
    NumcodecsCodec = None  # type: ignore[assignment]
    numcodecs_get_codec = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ArrayInfo:
    shape: Tuple[int, ...]
    dtype: np.dtype
    fortran_order: bool


class LmdbNpyStore:
    """
    LMDB-backed store for arbitrary NumPy arrays using .npy bytes as payload.

    - Keys: Python str (stored as UTF-8 bytes with a prefix)
    - Values: self-describing blob (dtype + shape are inside .npy header)
    - Compression: optional (numcodecs-backed) compression on the .npy bytes

    Notes:
    - This forbids dtype=object (no pickles) for safety/portability.
    - Fixed-width unicode (dtype kind 'U') is supported.
    """

    # Process-level env cache: resolved path -> (env, refcount).
    # LMDB forbids opening the same path twice in one process; reuse the env.
    _ENV_CACHE: dict[str, tuple["lmdb.Environment", int]] = {}

    _MAGIC = b"ANPY"
    _FMT_VERSION = 2  # record-format version for new writes
    _KEY_PREFIX = b"a:"  # namespace prefix for user arrays

    # v1 header: MAGIC(4) | fmt_ver(u8=1) | codec_cfg_len(u16) | raw_len(u32) | cfg | payload
    _HDR_STRUCT_V1 = struct.Struct(">4sBHI")

    # v2 header: MAGIC(4) | fmt_ver(u8=2) | codec_cfg_len(u16) | body_raw_len(u32)
    #          | npyhdr_len(u16) | cfg | npy_header | codec(body)
    # The .npy header stays uncompressed so get_info() can read shape/dtype
    # without invoking the codec.
    _HDR_STRUCT_V2 = struct.Struct(">4sBHIH")

    def __init__(
        self,
        path: str,
        *,
        map_size: int = 1 << 40,  # 1 TB sparse ceiling
        codec: Optional[Any] = None,
        readonly: bool = False,
        no_writers: bool = False,
        max_dbs: int = 1,
        max_readers: int = 256,
        readahead: bool = True,
        subdir: bool = True,
        lock: Optional[bool] = None,
        sync: bool = True,
        metasync: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        path:
            Directory (default LMDB style) or file path (if subdir=False).
        map_size:
            Maximum size of the database. LMDB maps are sparse on 64-bit
            systems (virtual memory, not committed disk), so defaulting to a
            large value is cheap. Must be >= the total bytes this store will
            ever hold. To grow later, call :meth:`set_map_size` while no
            transactions are open.
        codec:
            None (no compression), a numcodecs Codec instance, or legacy strings
            "none"/"zlib" for compatibility. For full functionality, install
            numcodecs and pass codec objects (mirrors Zarr API).
        readonly:
            Open environment read-only.
        no_writers:
            Caller's promise that no process will write to this env while it
            is open. When combined with ``readonly=True``, file locking is
            skipped for faster many-reader opens. Ignored if
            ``readonly=False``.
        lock:
            If None (default), locking is enabled unless
            (``readonly`` and ``no_writers``). Pass True/False to override.
        sync, metasync:
            LMDB durability flags. Defaults (True) fsync on every commit —
            safe but slow for many small write txns. Set both False for
            regenerable/resumable data: no per-commit fsync (OS flushes
            lazily), DB stays consistent on a process crash; only a system
            power loss can drop the most recent txns. NOTE: these are honored
            only on the FIRST open of a given path within a process (envs are
            cached by path); a later open with different flags reuses the
            cached env.
        """
        codec = self._normalize_codec(codec)

        if lock is None:
            # Safe rule: skip locking only when we guarantee no writer exists.
            lock = not (readonly and no_writers)

        if subdir and not readonly:
            os.makedirs(path, exist_ok=True)

        self._codec: Optional[Any] = codec
        self._map_size: int = map_size

        resolved = str(os.path.realpath(path))
        if resolved in LmdbNpyStore._ENV_CACHE:
            env, refcount = LmdbNpyStore._ENV_CACHE[resolved]
            LmdbNpyStore._ENV_CACHE[resolved] = (env, refcount + 1)
        else:
            env = lmdb.open(
                str(path),
                map_size=map_size,
                subdir=subdir,
                readonly=readonly,
                create=not readonly,
                lock=lock,
                max_dbs=max_dbs,
                max_readers=max_readers,
                readahead=readahead,
                sync=sync,
                metasync=metasync,
            )
            LmdbNpyStore._ENV_CACHE[resolved] = (env, 1)

        self._env = env
        self._resolved_path = resolved
        self._db = self._env.open_db(b"arrays")

    def close(self) -> None:
        env, refcount = LmdbNpyStore._ENV_CACHE.get(self._resolved_path, (self._env, 1))
        if refcount <= 1:
            LmdbNpyStore._ENV_CACHE.pop(self._resolved_path, None)
            self._env.close()
        else:
            LmdbNpyStore._ENV_CACHE[self._resolved_path] = (env, refcount - 1)

    def __enter__(self) -> "LmdbNpyStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def set_map_size(self, n: int) -> None:
        """
        Resize the LMDB map to ``n`` bytes.

        Caller must ensure no transactions are open in this process when
        called (LMDB constraint on ``mdb_env_set_mapsize``).
        """
        self._env.set_mapsize(n)
        self._map_size = n

    @classmethod
    def _encode_key(cls, key: str) -> bytes:
        if not isinstance(key, str):
            raise TypeError("key must be a str")
        b = key.encode("utf-8")
        return cls._KEY_PREFIX + b

    @classmethod
    def _decode_key(cls, raw: bytes) -> str:
        if not raw.startswith(cls._KEY_PREFIX):
            # internal/system keys could exist; ignore them in keys()
            raise ValueError("not a user key")
        return raw[len(cls._KEY_PREFIX) :].decode("utf-8")

    @staticmethod
    def _normalize_codec(codec: Optional[Any]) -> Optional[Any]:
        if codec is None:
            return None
        if isinstance(codec, str):
            lowered = codec.lower()
            if lowered == "none":
                return None
            if lowered == "zlib":
                return "zlib"
            if numcodecs_get_codec is None:
                raise ValueError(
                    f"Codec '{codec}' requires numcodecs; install numcodecs or pass a codec object."
                )
            return numcodecs_get_codec({"id": lowered})
        if NumcodecsCodec is not None and isinstance(codec, NumcodecsCodec):
            return codec
        raise TypeError(
            "codec must be None, a numcodecs Codec instance, or 'none'/'zlib' for compatibility."
        )

    def _encode_body(self, body: bytes) -> Tuple[bytes, bytes]:
        """Encode *body* with the configured codec. Returns (cfg_bytes, payload)."""
        if self._codec is None:
            return b"", body
        if NumcodecsCodec is not None and isinstance(self._codec, NumcodecsCodec):
            config = self._codec.get_config()
            cfg_bytes = json.dumps(
                config, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
            payload_obj = self._codec.encode(body)
            payload = (
                payload_obj
                if isinstance(payload_obj, (bytes, bytearray))
                else bytes(payload_obj)
            )
            return cfg_bytes, payload
        if self._codec == "zlib":
            return b'{"id":"zlib"}', zlib.compress(body)
        raise TypeError(
            "Unsupported codec: expected None, 'zlib', or a numcodecs Codec instance."
        )

    @staticmethod
    def _decode_body(cfg_bytes: bytes, encoded: bytes) -> bytes:
        """Inverse of :meth:`_encode_body`. Empty cfg_bytes means pass-through."""
        if not cfg_bytes:
            return encoded
        config = json.loads(cfg_bytes.decode("utf-8"))
        codec_id = config.get("id")
        if codec_id in (None, "none"):
            return encoded
        if codec_id == "zlib":
            return zlib.decompress(encoded)
        if numcodecs_get_codec is None:
            raise ImportError(
                "numcodecs is required to decode this record (codec config present)."
            )
        codec = numcodecs_get_codec(config)
        payload_obj = codec.decode(encoded)
        return (
            payload_obj
            if isinstance(payload_obj, (bytes, bytearray))
            else bytes(payload_obj)
        )

    @staticmethod
    def _split_npy(npy: bytes) -> Tuple[bytes, bytes]:
        """Split raw .npy bytes into (header_bytes, body_bytes) at the array
        data offset. The returned header is self-describing (magic + dict)."""
        bio = io.BytesIO(npy)
        version = npfmt.read_magic(bio)
        if version == (1, 0):
            npfmt.read_array_header_1_0(bio)
        elif version == (2, 0):
            npfmt.read_array_header_2_0(bio)
        elif version == (3, 0):
            npfmt.read_array_header_3_0(bio)
        else:
            raise ValueError(f"unsupported .npy version: {version}")
        offset = bio.tell()
        return npy[:offset], npy[offset:]

    @classmethod
    def _parse_record_header(
        cls, blob: bytes
    ) -> Tuple[int, int, int, int, int]:
        """Return (fmt_ver, cfg_len, raw_len, npyhdr_len, header_size).

        ``npyhdr_len`` is 0 for v1 records; ``raw_len`` is the length of the
        decoded body (v2) or the full decoded .npy blob (v1)."""
        if len(blob) < 5:
            raise ValueError("corrupt value (too small)")
        magic = blob[:4]
        fmt_ver = blob[4]
        if magic != cls._MAGIC:
            raise ValueError("corrupt value (bad magic)")
        if fmt_ver == 1:
            if len(blob) < cls._HDR_STRUCT_V1.size:
                raise ValueError("corrupt value (too small for v1 header)")
            _, _, cfg_len, raw_len = cls._HDR_STRUCT_V1.unpack_from(blob, 0)
            return fmt_ver, cfg_len, raw_len, 0, cls._HDR_STRUCT_V1.size
        if fmt_ver == 2:
            if len(blob) < cls._HDR_STRUCT_V2.size:
                raise ValueError("corrupt value (too small for v2 header)")
            _, _, cfg_len, raw_len, npyhdr_len = cls._HDR_STRUCT_V2.unpack_from(
                blob, 0
            )
            return fmt_ver, cfg_len, raw_len, npyhdr_len, cls._HDR_STRUCT_V2.size
        raise ValueError(f"unsupported record format version: {fmt_ver}")

    def _pack_value(self, npy_bytes: bytes) -> bytes:
        npy_header, body = self._split_npy(npy_bytes)
        cfg_bytes, encoded = self._encode_body(body)
        if len(cfg_bytes) > 0xFFFF:
            raise ValueError("codec config too large to store")
        if len(npy_header) > 0xFFFF:
            raise ValueError(".npy header too large to store in a v2 record")
        header = self._HDR_STRUCT_V2.pack(
            self._MAGIC,
            self._FMT_VERSION,
            len(cfg_bytes),
            len(body),
            len(npy_header),
        )
        return header + cfg_bytes + npy_header + encoded

    def _unpack_value(self, blob: bytes) -> bytes:
        fmt_ver, cfg_len, raw_len, npyhdr_len, hdr_size = self._parse_record_header(
            blob
        )
        off = hdr_size
        if len(blob) < off + cfg_len:
            raise ValueError("corrupt value (incomplete codec config)")
        cfg_bytes = blob[off : off + cfg_len]
        off += cfg_len

        if fmt_ver == 1:
            # v1: the whole encoded payload is the full .npy stream.
            payload = blob[off:]
            npy = self._decode_body(cfg_bytes, payload)
            if len(npy) != raw_len:
                raise ValueError("corrupt value (raw length mismatch)")
            return npy

        # v2: uncompressed .npy header + separately encoded body.
        if len(blob) < off + npyhdr_len:
            raise ValueError("corrupt value (incomplete .npy header)")
        npy_header = blob[off : off + npyhdr_len]
        off += npyhdr_len
        encoded_body = blob[off:]
        body = self._decode_body(cfg_bytes, encoded_body)
        if len(body) != raw_len:
            raise ValueError("corrupt value (body length mismatch)")
        return npy_header + body

    def _read_info_only(self, blob: bytes) -> "ArrayInfo":
        """Read shape/dtype without running the codec when possible (v2)."""
        fmt_ver, cfg_len, _, npyhdr_len, hdr_size = self._parse_record_header(blob)
        if fmt_ver == 2:
            off = hdr_size + cfg_len
            if len(blob) < off + npyhdr_len:
                raise ValueError("corrupt value (incomplete .npy header)")
            return self._read_npy_header(blob[off : off + npyhdr_len])
        # v1 stores codec over the whole .npy; no shortcut.
        return self._read_npy_header(self._unpack_value(blob))

    @staticmethod
    def _to_npy_bytes(arr: np.ndarray) -> bytes:
        arr = np.asarray(arr)
        if arr.dtype == object or arr.dtype.kind == "O":
            raise TypeError("dtype=object is not supported (would require pickling).")
        # Fixed-width unicode is fine (dtype.kind == 'U').
        bio = io.BytesIO()
        # allow_pickle=False: safe, deterministic
        np.save(bio, arr, allow_pickle=False)
        return bio.getvalue()

    @staticmethod
    def _from_npy_bytes(npy: bytes) -> np.ndarray:
        bio = io.BytesIO(npy)
        return np.load(bio, allow_pickle=False)

    @staticmethod
    def _read_npy_header(npy: bytes) -> ArrayInfo:
        bio = io.BytesIO(npy)
        version = npfmt.read_magic(bio)  # (major, minor)
        if version == (1, 0):
            shape, fortran_order, dtype = npfmt.read_array_header_1_0(bio)
        elif version == (2, 0):
            shape, fortran_order, dtype = npfmt.read_array_header_2_0(bio)
        elif version == (3, 0):
            # numpy may emit 3.0 for large headers
            shape, fortran_order, dtype = npfmt.read_array_header_3_0(bio)
        else:
            raise ValueError(f"unsupported .npy version: {version}")
        return ArrayInfo(tuple(shape), np.dtype(dtype), bool(fortran_order))

    # ---------- public API ----------

    def put(self, key: str, array: np.ndarray) -> None:
        k = self._encode_key(key)
        npy = self._to_npy_bytes(array)
        v = self._pack_value(npy)
        with self._env.begin(write=True, db=self._db) as txn:
            txn.put(k, v, overwrite=True)

    def put_many(self, items) -> None:
        """Write many (key, array) pairs in a single write transaction.

        Far cheaper than repeated :meth:`put` (one txn commit instead of N).
        """
        with self._env.begin(write=True, db=self._db) as txn:
            for key, array in items:
                k = self._encode_key(key)
                v = self._pack_value(self._to_npy_bytes(array))
                txn.put(k, v, overwrite=True)

    def get(self, key: str) -> np.ndarray:
        k = self._encode_key(key)
        with self._env.begin(write=False, db=self._db) as txn:
            blob = txn.get(k)
        if blob is None:
            raise KeyError(key)
        npy = self._unpack_value(blob)
        return self._from_npy_bytes(npy)

    def get_info(self, key: str) -> ArrayInfo:
        k = self._encode_key(key)
        with self._env.begin(write=False, db=self._db) as txn:
            blob = txn.get(k)
        if blob is None:
            raise KeyError(key)
        return self._read_info_only(blob)

    def delete(self, key: str) -> bool:
        k = self._encode_key(key)
        with self._env.begin(write=True, db=self._db) as txn:
            return bool(txn.delete(k))

    def keys(self) -> Iterator[str]:
        with self._env.begin(write=False, db=self._db) as txn:
            with txn.cursor() as cur:
                if not cur.set_range(self._KEY_PREFIX):
                    return
                for raw_key in cur.iternext(keys=True, values=False):
                    if not raw_key.startswith(self._KEY_PREFIX):
                        break
                    yield self._decode_key(raw_key)

    def values(self) -> Iterator[NDArray]:
        with self._env.begin(write=False, db=self._db) as txn:
            with txn.cursor() as cur:
                if not cur.set_range(self._KEY_PREFIX):
                    return
                for raw_key, blob in cur.iternext(keys=True, values=True):
                    if not raw_key.startswith(self._KEY_PREFIX):
                        break
                    npy = self._unpack_value(blob)
                    yield self._from_npy_bytes(npy)

    def count(self) -> int:
        with self._env.begin(write=False, db=self._db) as txn:
            with txn.cursor() as cur:
                if not cur.set_range(self._KEY_PREFIX):
                    return 0
                n = 0
                for raw_key in cur.iternext(keys=True, values=False):
                    if not raw_key.startswith(self._KEY_PREFIX):
                        break
                    n += 1
                return n
