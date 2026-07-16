"""
Custom codecs for LMDB-backed numpy storage.

Currently includes:
- PngCodec: stores a batch of images (N, H, W, C) as concatenated PNGs.
"""

from __future__ import annotations

import io
import struct
from typing import Any, List

import numpy as np

try:  # optional dependency
    from PIL import Image
except ImportError:  # pragma: no cover - exercised only when Pillow missing
    Image = None  # type: ignore[assignment]

try:
    from numcodecs.abc import Codec as NumcodecsCodec
    from numcodecs.registry import register_codec as numcodecs_register_codec
except ImportError:  # pragma: no cover - optional dependency
    NumcodecsCodec = object  # sentinel to allow isinstance checks to fail
    numcodecs_register_codec = None  # type: ignore[assignment]


class PngCodec(NumcodecsCodec):  # type: ignore[misc]
    """
    Encode a batch of images (N, H, W, C) into concatenated PNG bytes.

    - C must be 1 (L), 3 (RGB), or 4 (RGBA).
    - dtype must be uint8.
    - During encoding each image is converted to PNG; payload layout:
        [count:u32][len_0:u32 ... len_{N-1}:u32][png0][png1]...[png_{N-1}]
    """

    codec_id = "png"
    _LEN_STRUCT = struct.Struct(">I")  # big-endian u32 for counts/lengths

    def __init__(self) -> None:
        if Image is None:
            raise ImportError("Pillow is required for PngCodec (pip install pillow).")

    def encode(self, buf: Any) -> bytes:
        arr = np.asarray(buf)
        if arr.ndim != 4:
            raise ValueError("PngCodec expects array with shape (N, H, W, C).")
        n, h, w, c = arr.shape
        if c not in (1, 3, 4):
            raise ValueError("PngCodec expects C in {1, 3, 4}.")
        if arr.dtype != np.uint8:
            raise TypeError("PngCodec only supports dtype=uint8.")
        if n == 0:
            # Only header; no payload
            return self._LEN_STRUCT.pack(0)

        frames: List[bytes] = []
        for i in range(n):
            frame = arr[i]
            if c == 1:
                # drop the singleton channel for PIL "L" mode
                frame_img = Image.fromarray(frame[..., 0], mode="L")
            elif c == 3:
                frame_img = Image.fromarray(frame, mode="RGB")
            else:  # c == 4
                frame_img = Image.fromarray(frame, mode="RGBA")

            buf_io = io.BytesIO()
            frame_img.save(buf_io, format="PNG")
            frames.append(buf_io.getvalue())

        header = [
            self._LEN_STRUCT.pack(n),
            b"".join(self._LEN_STRUCT.pack(len(f)) for f in frames),
        ]
        return b"".join(header + frames)

    def decode(self, buf: Any, out: Any = None) -> np.ndarray:  # type: ignore[override]
        if Image is None:
            raise ImportError("Pillow is required for PngCodec (pip install pillow).")
        data = memoryview(buf).tobytes() if isinstance(buf, memoryview) else bytes(buf)
        if len(data) < self._LEN_STRUCT.size:
            raise ValueError("corrupt PNG payload (too small for count header)")

        offset = 0
        (n,) = self._LEN_STRUCT.unpack_from(data, offset)
        offset += self._LEN_STRUCT.size

        lengths: List[int] = []
        for _ in range(n):
            if offset + self._LEN_STRUCT.size > len(data):
                raise ValueError("corrupt PNG payload (truncated lengths header)")
            (ln,) = self._LEN_STRUCT.unpack_from(data, offset)
            lengths.append(ln)
            offset += self._LEN_STRUCT.size

        images: List[np.ndarray] = []
        for ln in lengths:
            end = offset + ln
            if end > len(data):
                raise ValueError("corrupt PNG payload (truncated frame)")
            frame_bytes = data[offset:end]
            offset = end
            with Image.open(io.BytesIO(frame_bytes)) as img:
                img = img.convert("RGBA") if img.mode == "RGBA" else img.convert("RGB" if img.mode != "L" else "L")
                frame_arr = np.asarray(img)
                if frame_arr.ndim == 2:  # L mode
                    frame_arr = frame_arr[..., None]
                images.append(frame_arr)

        if len(images) == 0:
            # No frames; return empty batch with shape (0, 0, 0, 0) to be explicit
            return np.empty((0, 0, 0, 0), dtype=np.uint8)

        # Ensure consistent channel dimension (1, 3, or 4) based on first frame
        c = images[0].shape[-1]
        for i, img in enumerate(images):
            if img.shape[-1] != c:
                raise ValueError(f"inconsistent channel counts in decoded frames (frame {i})")

        return np.stack(images, axis=0).astype(np.uint8, copy=False)

    def get_config(self) -> dict:
        return {"id": self.codec_id}

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "PngCodec()"


def register_png_codec() -> None:
    """Register PngCodec with numcodecs (no-op if numcodecs unavailable)."""
    if numcodecs_register_codec is None:
        raise ImportError("numcodecs is required to register PngCodec.")
    numcodecs_register_codec(PngCodec)


__all__ = ["PngCodec", "register_png_codec"]
