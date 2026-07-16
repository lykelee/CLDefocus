from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from datasyn.synthesis.providers import Photo, PhotoProvider, PhotoProviderSpec
from datasyn.utils.files import get_files_by_names


def all_sets_equal(*collections: Iterable) -> bool:
    """Return True if all input collections have identical set representations."""
    iterator = iter(collections)

    try:
        first = set(next(iterator))
    except StopIteration:
        # No inputs -> vacuously true
        return True

    return all(set(col) == first for col in iterator)


class NormalPhoto(Photo):
    def __init__(
        self,
        linear: np.ndarray,
        depth: np.ndarray,
        paired_blurry_rgb: np.ndarray | None = None,
    ) -> None:
        self._linear = linear
        self._depth = depth
        self._paired_blurry_rgb = paired_blurry_rgb

    def get_linear(self) -> np.ndarray:
        return self._linear

    def get_depth(self) -> np.ndarray:
        return self._depth

    def lin_to_rgb(self, lin: np.ndarray) -> np.ndarray:
        return np.clip(lin, 0.0, 1.0) ** (1 / 2.2)

    def raw_to_lin(self, raw: np.ndarray) -> np.ndarray:
        """Demosaic RGGB Bayer -> linear RGB (H, W, 3), float32 [0, 1].

        Uses the same demosaic algorithm as EasyRaw (menon2007 via pipeline_utils).
        """
        from datasyn.isp import pipeline_utils as pu

        raw = np.asarray(raw, dtype=np.float32)
        # raw = np.clip(raw, 0.0, 1.0)
        return pu.demosaic(
            raw,
            cfa_pattern=[0, 1, 1, 2],
            output_channel_order="RGB",
            alg_type="EA",
            # alg_type="menon2007",
        ).astype(np.float32)

    def linear_to_raw(self, x: np.ndarray) -> np.ndarray:
        """Virtual Bayer mosaicing (RGGB): linear (H, W, 3) -> normalized 2D CFA (H, W).

        Uses the same mosaic function as EasyRaw (datasyn.isp.mosaic.mosaic_bayer).
        Since Hypersim has no real sensor, RAW space is defined synthetically.
        Output is in [0, 1].
        """
        from datasyn.isp.mosaic import mosaic_bayer

        x = np.asarray(x, dtype=np.float32)
        # x = np.clip(x, 0.0, 1.0)
        return mosaic_bayer(x, pattern=[0, 1, 1, 2])

    def get_paired_blurry_rgb(self) -> np.ndarray:
        if self._paired_blurry_rgb is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no paired blurry image configured."
            )
        return self._paired_blurry_rgb


class NormalPhotoProvider(PhotoProvider):
    def __init__(
        self,
        dir_raw: Path,
        dir_rgb: Path,
        dir_dpt: Path,
        dir_blurry_raw: Path = None,
        sharp_to_blurry: dict[str, str] | None = None,
    ) -> None:
        self._dir_raw = Path(dir_raw) if dir_raw else None
        self._dir_rgb = Path(dir_rgb) if dir_rgb else None
        self._dir_dpt = Path(dir_dpt)
        self._dir_blurry_raw = Path(dir_blurry_raw) if dir_blurry_raw else None
        self._sharp_to_blurry = dict(sharp_to_blurry or {})

        name_sets: set[str] = []

        dpts = get_files_by_names(
            dir_dpt,
            pred=lambda x: x.suffix.lower() == ".npz",
            remove_exts=True,
        )
        name_sets.append(set(dpts.keys()))

        if dir_raw is not None:
            raws = get_files_by_names(
                dir_raw,
                pred=lambda x: x.suffix.lower() in [".cr2", ".dng"],
                remove_exts=True,
            )
            rgbs = None
            if self._dir_blurry_raw is not None:
                blurry_raws = get_files_by_names(
                    self._dir_blurry_raw,
                    pred=lambda x: x.suffix.lower() in [".cr2", ".dng"],
                    remove_exts=True,
                )
            else:
                blurry_raws = None
        else:
            raws = None
            rgbs = get_files_by_names(
                dir_rgb, pred=lambda x: x.suffix == ".png", remove_exts=True
            )
            blurry_raws = None

        names = sorted(set.intersection(*name_sets))
        self._names = names
        self._raws = raws
        self._rgbs = rgbs
        self._blurry_raws = blurry_raws
        self._dpts = dpts

    def get_names(self) -> list[str]:
        return list(self._names)

    def open(self, photo: str):
        from PIL import Image

        from datasyn.utils.img.convert import pil_to_arr1f

        file_dpt = self._dpts[photo]

        if self._raws is not None:
            from datasyn.synthesis.providers.raw import RawPhoto

            file_raw = self._raws[photo]
            if self._blurry_raws is not None:
                if photo not in self._sharp_to_blurry:
                    raise KeyError(
                        f"No sharp_to_blurry mapping for sharp photo '{photo}'"
                    )
                blurry_name = self._sharp_to_blurry[photo]
                try:
                    file_blurry_raw = self._blurry_raws[blurry_name]
                except KeyError as e:
                    raise KeyError(
                        f"Mapped blurry RAW '{blurry_name}' not found for sharp photo '{photo}'"
                    ) from e
            else:
                file_blurry_raw = None

            return RawPhoto(
                raw_path=file_raw,
                depth_path=file_dpt,
                blurry_raw_path=file_blurry_raw,
            )

        else:
            file_rgb = self._rgbs[photo]
            rgb = pil_to_arr1f(Image.open(file_rgb), dtype=np.float32)
            linear = rgb**2.2

            depth = np.asarray(np.load(file_dpt)["depth"], dtype=np.float32)

            return NormalPhoto(linear, depth)


@dataclass
class NormalProviderSpec(PhotoProviderSpec):
    dir_raw: Path = None
    dir_rgb: Path = None
    dir_dpt: Path = None
    dir_blurry_raw: Path = None
    sharp_to_blurry: dict[str, str] | None = None

    def build(self):
        return NormalPhotoProvider(
            dir_raw=self.dir_raw,
            dir_rgb=self.dir_rgb,
            dir_dpt=self.dir_dpt,
            dir_blurry_raw=self.dir_blurry_raw,
            sharp_to_blurry=self.sharp_to_blurry,
        )
