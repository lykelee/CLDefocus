"""
Photo and PhotoProvider backed by RAW files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from datasyn.synthesis.providers import Photo, PhotoProvider, PhotoProviderSpec


def _load_depth(path: Path) -> np.ndarray:
    """Load a depth map from .npz (key='depth') or .npy."""
    if path.suffix.lower() == ".npz":
        return np.load(path)["depth"]
    return np.load(path)


class RawPhoto(Photo):
    """
    Photo handle backed by a single RAW file.

    EasyRaw is opened at construction. Depth and the resized linear image are
    loaded lazily on first access so that Stage 6 (s6_blur) (which only needs
    linear_to_raw) avoids unnecessary I/O.
    """

    def __init__(
        self,
        raw_path: Path,
        depth_path: Path,
        blurry_raw_path: Path | None = None,
        depth_clip: tuple[float, float] = (0.1, 250.0),
    ) -> None:
        from datasyn.isp.easyraw import EasyRaw

        self._raw = EasyRaw(raw_path)
        self._blurry_raw_path = blurry_raw_path
        self._depth_path = depth_path
        self._depth_clip = depth_clip

        self._depth: Optional[np.ndarray] = None
        self._linear: Optional[np.ndarray] = None
        self._paired_blurry_rgb: Optional[np.ndarray] = None

    # ── Photo interface ────────────────────────────────────────────────────────

    def get_linear(self) -> np.ndarray:
        if self._linear is None:
            depth = self.get_depth()
            H, W = int(depth.shape[0]), int(depth.shape[1])
            import cv2

            img_large = self._raw.get_source_linear().astype(np.float32)
            # cv2 INTER_AREA is the correct, fast antialiasing downsampler.
            # (Replaces a slow JAX LANCZOS3 CPU resize of the full-res RAW.)
            self._linear = cv2.resize(
                np.ascontiguousarray(img_large), (W, H), interpolation=cv2.INTER_AREA
            ).astype(np.float32)
        return self._linear

    def get_depth(self) -> np.ndarray:
        if self._depth is None:
            depth = _load_depth(self._depth_path)
            self._depth = np.clip(
                depth, self._depth_clip[0], self._depth_clip[1]
            ).astype(np.float32)
        return self._depth

    def lin_to_rgb(self, lin: np.ndarray) -> np.ndarray:
        return np.asarray(self._raw.linear_to_rgb(lin))

    def raw_to_lin(self, raw: np.ndarray) -> np.ndarray:
        from datasyn.isp.easyraw import EasyRawConfig

        return np.asarray(
            self._raw.raw_to_linear(np.asarray(raw), config=EasyRawConfig())
        )

    def linear_to_raw(self, x: np.ndarray) -> np.ndarray:
        from datasyn.isp.easyraw import EasyRawConfig

        return np.asarray(
            self._raw.linear_to_raw(np.asarray(x), config=EasyRawConfig())
        )

    def get_paired_blurry_rgb(self) -> np.ndarray:
        if self._blurry_raw_path is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no paired blurry RAW configured."
            )
        if self._paired_blurry_rgb is None:
            import jax.image as jimg
            import jax.numpy as jnp

            from datasyn.isp.easyraw import EasyRaw

            depth = self.get_depth()
            H, W = int(depth.shape[0]), int(depth.shape[1])
            blurry_raw = EasyRaw(self._blurry_raw_path)
            blurry_lin_large = blurry_raw.get_source_linear().astype(np.float32)
            blurry_lin = np.asarray(
                jimg.resize(
                    jnp.asarray(blurry_lin_large),
                    (H, W, 3),
                    method=jimg.ResizeMethod.LANCZOS3,
                )
            )
            self._paired_blurry_rgb = np.asarray(blurry_raw.linear_to_rgb(blurry_lin))
        return self._paired_blurry_rgb


class RawPhotoProvider(PhotoProvider):
    """PhotoProvider backed by a DefocusDataset (RAW + depth files)."""

    def __init__(
        self,
        dataset,  # DefocusDataset — typed loosely to avoid circular import
        blurry_raws: dict[str, Path] | None = None,
        depth_clip: tuple[float, float] = (0.1, 250.0),
    ) -> None:
        self._dataset = dataset
        self._blurry_raws = blurry_raws or {}
        self._depth_clip = depth_clip

    def get_names(self) -> list[str]:
        return self._dataset.get_names()

    def open(self, photo: str) -> RawPhoto:
        return RawPhoto(
            self._dataset.get_raw(photo),
            self._dataset.get_depth(photo),
            blurry_raw_path=self._blurry_raws.get(photo),
            depth_clip=self._depth_clip,
        )


@dataclass
class RawProviderSpec(PhotoProviderSpec):
    """Picklable spec for RawPhotoProvider backed by FilesystemDefocusDataset."""

    raw_dir: Path
    depth_dir: Path
    blurry_raw_dir: Path | None = None
    sharp_to_blurry: dict[str, str] | None = None
    raw_suffix: str = ".CR2"
    depth_suffix: str = ".npz"
    blurry_raw_suffix: str = ".CR2"
    depth_clip: tuple[float, float] = (0.1, 250.0)

    def build(self) -> RawPhotoProvider:
        from datasyn.synthesis.dataset import FilesystemDefocusDataset
        from datasyn.utils.files import get_files_by_names

        blurry_raws: dict[str, Path] | None = None
        if self.blurry_raw_dir is not None:
            if self.sharp_to_blurry is None:
                raise ValueError(
                    "RawProviderSpec requires sharp_to_blurry when blurry_raw_dir is set."
                )
            blurry_files = get_files_by_names(
                self.blurry_raw_dir,
                pred=lambda x: x.suffix.lower() == self.blurry_raw_suffix.lower(),
                remove_exts=True,
            )
            missing = sorted(
                set(self.sharp_to_blurry.values()) - set(blurry_files.keys())
            )
            if missing:
                raise KeyError(
                    "Missing blurry RAW files for mapped names: " + ", ".join(missing)
                )
            blurry_raws = {
                sharp_name: blurry_files[blurry_name]
                for sharp_name, blurry_name in self.sharp_to_blurry.items()
            }

        return RawPhotoProvider(
            FilesystemDefocusDataset(
                raw_dir=self.raw_dir,
                depth_dir=self.depth_dir,
                raw_suffix=self.raw_suffix,
                depth_suffix=self.depth_suffix,
            ),
            blurry_raws=blurry_raws,
            depth_clip=self.depth_clip,
        )
