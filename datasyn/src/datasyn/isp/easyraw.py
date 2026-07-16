"""
Convenient raw image manipulation for image synthesis.
"""

import os
from dataclasses import dataclass
from typing import Optional

from numpy.typing import NDArray

from datasyn.isp import pipeline_utils as pu

from .builder import build_default_pipeline
from .core import Pipeline


@dataclass
class EasyRawConfig:
    """Configuration overrides for EasyRaw pipeline operations."""

    cm: Optional[NDArray] = None
    tc: bool = False
    demosaic_type: str = "menon2007"


class EasyRaw:
    """
    High-level interface for raw image manipulation, useful for degradation modeling.
    """

    def __init__(
        self,
        rawfile: os.PathLike,
        demosaic_type: str = "menon2007",
        apply_orientation: bool = True,
    ):
        self.raw_image, self.raw = pu.get_visible_raw_image(rawfile)
        self.metadata = pu.get_metadata(rawfile, self.raw)
        self.pipeline: Pipeline = build_default_pipeline(
            demosaic_type=demosaic_type, apply_orientation=apply_orientation
        )
        self.default_demosaic_type = demosaic_type

    @property
    def white_level(self) -> int:
        return self.metadata["white_level"]

    def run_pipeline(
        self,
        image: NDArray,
        *,
        input_stage: str,
        output_stage: str,
        metadata: Optional[dict] = None,
        direction: str = "forward",
        config: Optional[EasyRawConfig] = None,
    ) -> NDArray:
        """Run pipeline between two stages with optional config overrides."""
        config = config or EasyRawConfig()

        stage_overrides = {
            "demosaic": {"demosaic_type": config.demosaic_type},
        }

        if config.cm is not None:
            stage_overrides["xyz"] = {"cm": config.cm}

        if config.tc:
            stage_overrides["tone"] = {"tc": config.tc}

        return self.pipeline.run(
            image,
            metadata or self.metadata,
            input_stage=input_stage,
            output_stage=output_stage,
            config_overrides=stage_overrides,
            direction=direction,
        )

    def get_source_linear(self) -> NDArray:
        """Get source image as linear RGB (raw -> srgb stage)."""
        return self.run_pipeline(self.raw_image, input_stage="raw", output_stage="srgb")

    def raw_to_linear(
        self, image: NDArray, config: Optional[EasyRawConfig] = None
    ) -> NDArray:
        """Convert normalized raw (Bayer) to linear RGB."""
        return self.run_pipeline(
            image, input_stage="normal", output_stage="srgb", config=config
        )

    def linear_to_raw(
        self, image: NDArray, config: Optional[EasyRawConfig] = None
    ) -> NDArray:
        """Convert linear RGB to normalized raw (Bayer) - inverse, lossy."""
        return self.run_pipeline(
            image,
            input_stage="srgb",
            output_stage="normal",
            direction="inverse",
            config=config,
        )

    def linear_to_rgb(
        self, image: NDArray, config: Optional[EasyRawConfig] = None
    ) -> NDArray:
        """Convert linear RGB to display RGB."""
        return self.run_pipeline(
            image, input_stage="srgb", output_stage="tone", config=config
        )

    def __repr__(self) -> str:
        h, w = self.raw_image.shape[:2]
        return (
            f"EasyRaw(shape=({h}, {w}), "
            f"cfa_pattern={self.metadata['cfa_pattern']}, "
            f"white_level={self.white_level})"
        )
