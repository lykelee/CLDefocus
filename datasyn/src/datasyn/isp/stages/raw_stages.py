"""
ISP stages operating on RAW/Bayer domain.
"""

import logging
from typing import Any, Mapping, Optional

import numpy as np
from numpy.typing import NDArray

from datasyn.isp import pipeline_utils as pu
from datasyn.isp.mosaic import mosaic_bayer

from ..metadata import get_as_shot_neutral, get_black_level, get_white_level
from .base import ConditionalStage, ValidatedStage

logger = logging.getLogger(__name__)


class NormalizeStage(ValidatedStage):
    """Normalize raw sensor values to [0, 1] via black/white level correction."""

    REQUIRED_METADATA_KEYS = ["black_level", "white_level"]

    def __init__(
        self,
        input_stage: str = "raw",
        output_stage: str = "normal",
        config: Mapping[str, Any] = None,
    ):
        super().__init__(input_stage, output_stage, config)

    def validate_metadata(self, metadata: Mapping[str, Any]) -> None:
        super().validate_metadata(metadata)
        if (
            "linearization_table" in metadata
            and metadata["linearization_table"] is not None
        ):
            logger.warning("Linearization table found but not currently supported")

    def forward(
        self, image: NDArray, metadata: Mapping[str, Any], config: Mapping[str, Any]
    ) -> NDArray:
        return pu.normalize(image, get_black_level(metadata), get_white_level(metadata))

    def inverse(
        self, image: NDArray, metadata: Mapping[str, Any], config: Mapping[str, Any]
    ) -> NDArray:
        black_level = metadata["black_level"]
        white_level = metadata["white_level"]

        if isinstance(black_level, list) and len(black_level) == 1:
            black_level = float(black_level[0])
        if isinstance(white_level, list) and len(white_level) == 1:
            white_level = float(white_level[0])

        black_level_mask = black_level
        if isinstance(black_level, list) and len(black_level) == 4:
            if isinstance(black_level[0], pu.Ratio):
                black_level = pu.ratios2floats(black_level)
            black_level_mask = np.zeros(image.shape)
            idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
            for i, idx in enumerate(idx2by2):
                black_level_mask[idx[0] :: 2, idx[1] :: 2] = black_level[i]

        return image * (white_level - black_level_mask) + black_level_mask


class LensShadingStage(ConditionalStage):
    """Apply lens shading correction using DNG opcode gain maps."""

    REQUIRED_METADATA_KEYS = ["cfa_pattern"]

    def __init__(
        self,
        input_stage: str = "normal",
        output_stage: str = "lens_shading_correction",
        config: Mapping[str, Any] = None,
    ):
        super().__init__(input_stage, output_stage, config)

    def _get_gain_map_opcode(self, metadata: Mapping[str, Any]) -> Optional[Any]:
        if "opcode_lists" not in metadata:
            return None
        if 51009 not in metadata["opcode_lists"]:
            return None
        opcode_list_2 = metadata["opcode_lists"][51009]
        if len(opcode_list_2) < 10:
            return None
        return opcode_list_2[9]

    def should_apply(self, metadata: Mapping[str, Any]) -> bool:
        return self._get_gain_map_opcode(metadata) is not None

    def forward_impl(
        self, image: NDArray, metadata: Mapping[str, Any], config: Mapping[str, Any]
    ) -> NDArray:
        raise NotImplementedError()

    def inverse_impl(
        self, image: NDArray, metadata: Mapping[str, Any], config: Mapping[str, Any]
    ) -> NDArray:
        gain_map_opcode = self._get_gain_map_opcode(metadata)
        gain_map = gain_map_opcode.data["map_gain_2d"]

        inv_gain_map = np.ones_like(gain_map)
        np.divide(1.0, gain_map, out=inv_gain_map, where=gain_map != 0)

        return pu.lens_shading_correction(
            image,
            gain_map_opcode=None,
            gain_map=inv_gain_map,
            bayer_pattern=metadata["cfa_pattern"],
            clip=config.get("clip", True),
        )


class WhiteBalanceStage(ValidatedStage):
    """Apply white balance correction using as-shot neutral values."""

    REQUIRED_METADATA_KEYS = ["as_shot_neutral", "cfa_pattern"]

    def __init__(
        self,
        input_stage: str = "lens_shading_correction",
        output_stage: str = "white_balance",
        config: Mapping[str, Any] = None,
    ):
        super().__init__(input_stage, output_stage, config)

    def forward(
        self, image: NDArray, metadata: Mapping[str, Any], config: Mapping[str, Any]
    ) -> NDArray:
        return pu.white_balance(
            image, metadata["as_shot_neutral"], metadata["cfa_pattern"]
        )

    def inverse(
        self, image: NDArray, metadata: Mapping[str, Any], config: Mapping[str, Any]
    ) -> NDArray:
        as_shot_neutral = get_as_shot_neutral(metadata)
        cfa_pattern = metadata["cfa_pattern"]

        balanced = np.zeros_like(image)
        idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
        for i, idx in enumerate(idx2by2):
            idx_y, idx_x = idx
            balanced[idx_y::2, idx_x::2] = (
                image[idx_y::2, idx_x::2] * as_shot_neutral[cfa_pattern[i]]
            )
        return balanced


class DemosaicStage(ValidatedStage):
    """Demosaic Bayer pattern to RGB."""

    REQUIRED_METADATA_KEYS = ["cfa_pattern"]

    def __init__(
        self,
        input_stage: str = "white_balance",
        output_stage: str = "demosaic",
        config: Mapping[str, Any] = None,
    ):
        default_config = {"demosaic_type": "menon2007", "output_channel_order": "RGB"}
        merged_config = {**default_config, **(config or {})}
        super().__init__(input_stage, output_stage, merged_config)

    def forward(
        self, image: NDArray, metadata: Mapping[str, Any], config: Mapping[str, Any]
    ) -> NDArray:
        return pu.demosaic(
            image,
            metadata["cfa_pattern"],
            output_channel_order=config.get("output_channel_order", "RGB"),
            alg_type=config.get("demosaic_type", "menon2007"),
        )

    def inverse(
        self, image: NDArray, metadata: Mapping[str, Any], config: Mapping[str, Any]
    ) -> NDArray:
        return mosaic_bayer(image, metadata["cfa_pattern"])
