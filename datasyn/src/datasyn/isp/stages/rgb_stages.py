"""
ISP stages operating on RGB/XYZ domain.
"""

import logging
import pathlib
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np
import scipy.io
from numpy.typing import NDArray

from datasyn.isp import pipeline_utils as pu
from datasyn.jaxutils import jx
from datasyn.jaxutils.typing import JArray

from ..core import Stage
from ..metadata import get_orientation
from .base import ValidatedStage

logger = logging.getLogger(__name__)

_MAT_DIR = pathlib.Path(__file__).parent.parent / "mat_collections"

M_lin2xyz_realblur = scipy.io.loadmat(_MAT_DIR / "M_lin2xyz.mat")["M"]
M_xyz2lin_realblur = scipy.io.loadmat(_MAT_DIR / "M_xyz2lin.mat")["M"]
M_cam2xyz_realblur = scipy.io.loadmat(_MAT_DIR / "a7r3_cam2xyz.mat")[
    "final_cam2xyz"
].astype("float32")
M_xyz2cam_realblur = scipy.io.loadmat(_MAT_DIR / "a7r3_xyz2cam.mat")[
    "final_xyz2cam"
].astype("float32")

alpha_realblur: NDArray = scipy.io.loadmat(_MAT_DIR / "a7r3_polynomial_alpha.mat")[
    "alpha"
].astype("float32")


@jx.jit
def _jax_lin2rgb_a7r3_polynomial(img: JArray) -> JArray:
    a0, a1, a2, a3, a4, a5, a6, a7 = (alpha_realblur[k][0] for k in range(8))
    srgb = (
        (((((a7 * img + a6) * img + a5) * img + a4) * img + a3) * img + a2) * img + a1
    ) * img + a0
    return jnp.clip(srgb, 0.0, 1.0)


class XYZStage(ValidatedStage):
    """Transform camera RGB to CIE XYZ color space."""

    REQUIRED_METADATA_KEYS = ["color_matrix_1", "color_matrix_2"]

    def __init__(
        self,
        input_stage: str = "demosaic",
        output_stage: str = "xyz",
        config: Mapping[str, Any] = None,
    ):
        super().__init__(input_stage, output_stage, config)

    def forward(
        self, image: NDArray, metadata: Mapping[str, Any], config: Mapping[str, Any]
    ) -> NDArray:
        cm1 = np.asarray(
            config.get("cm", metadata["color_matrix_1"]), dtype=np.float32
        ).reshape((3, 3))
        image = pu.apply_color_space_transform(image, cm1, metadata["color_matrix_2"])

        if config.get("apply_orientation", True):
            image = pu.fix_orientation(image, get_orientation(metadata))

        return image

    def inverse(
        self, image: NDArray, metadata: Mapping[str, Any], config: Mapping[str, Any]
    ) -> NDArray:
        if config.get("apply_orientation", True):
            image = pu.reverse_orientation(image, get_orientation(metadata))

        # NOTE: cm may be rational-valued; must cast to float32 to avoid object dtype.
        xyz2cam1 = np.asarray(
            config.get("cm", metadata["color_matrix_1"]), dtype=np.float32
        ).reshape((3, 3))
        xyz2cam1 = xyz2cam1 / np.sum(xyz2cam1, axis=1, keepdims=True)

        cam_rgb = xyz2cam1[np.newaxis, np.newaxis, :, :] * image[:, :, np.newaxis, :]
        return np.sum(cam_rgb, axis=-1)


class SRGBStage(ValidatedStage):
    """Transform XYZ to sRGB color space."""

    REQUIRED_METADATA_KEYS = []

    def __init__(
        self,
        input_stage: str = "xyz",
        output_stage: str = "srgb",
        config: Mapping[str, Any] = None,
    ):
        default_config = {"apply_orientation": True}
        merged_config = {**default_config, **(config or {})}
        super().__init__(input_stage, output_stage, merged_config)

    def forward(
        self, image: NDArray, metadata: Mapping[str, Any], config: Mapping[str, Any]
    ) -> NDArray:
        return pu.transform_xyz_to_srgb(image)

    def inverse(
        self, image: NDArray, metadata: Mapping[str, Any], config: Mapping[str, Any]
    ) -> NDArray:
        xyz2srgb = np.array(
            [
                [3.2404542, -1.5371385, -0.4985314],
                [-0.9692660, 1.8760108, 0.0415560],
                [0.0556434, -0.2040259, 1.0572252],
            ]
        )
        xyz2srgb = xyz2srgb / np.sum(xyz2srgb, axis=-1, keepdims=True)
        srgb2xyz = np.linalg.inv(xyz2srgb)

        xyz_image = srgb2xyz[np.newaxis, np.newaxis, :, :] * image[:, :, np.newaxis, :]
        return np.sum(xyz_image, axis=-1)


class RSBlurCRFStage(Stage):
    """Apply Sony A7R3 polynomial camera response function (linear -> display RGB)."""

    def __init__(
        self,
        input_stage: str = "srgb",
        output_stage: str = "tone",
        config: Mapping[str, Any] = None,
    ):
        default_config = {"kind": "a7r3_polynomial"}
        merged_config = {**default_config, **(config or {})}
        super().__init__(input_stage, output_stage, merged_config)

    def forward(
        self, image: NDArray, metadata: Mapping[str, Any], config: Mapping[str, Any]
    ) -> NDArray:
        kind = config.get("kind", "a7r3_polynomial")
        match kind:
            case "a7r3_polynomial":
                return _jax_lin2rgb_a7r3_polynomial(image)
            case _:
                raise NotImplementedError(f"Unknown CRF kind: {kind!r}")

    def inverse(
        self, image: NDArray, metadata: Mapping[str, Any], config: Mapping[str, Any]
    ) -> NDArray:
        raise NotImplementedError()
