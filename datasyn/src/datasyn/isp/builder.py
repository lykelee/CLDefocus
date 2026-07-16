"""
Pipeline builder for constructing ISP pipelines.

This module provides utilities for building pipelines with a fluent API
and factories for common pipeline configurations.
"""

from typing import List

from .core import Pipeline, Stage
from .stages.raw_stages import (
    DemosaicStage,
    LensShadingStage,
    NormalizeStage,
    WhiteBalanceStage,
)
from .stages.rgb_stages import RSBlurCRFStage, SRGBStage, XYZStage


class PipelineBuilder:
    """
    Builder for constructing ISP pipelines with a fluent API.

    Example:
        builder = PipelineBuilder()
        pipeline = (builder
            .add_stage(NormalizeStage())
            .add_stage(WhiteBalanceStage())
            .add_stage(DemosaicStage())
            .build())
    """

    def __init__(self):
        """Initialize empty pipeline builder."""
        self._stages: List[Stage] = []

    def add_stage(self, stage: Stage) -> "PipelineBuilder":
        """
        Add a stage to the pipeline.

        Args:
            stage: Stage instance to add

        Returns:
            Self for method chaining
        """
        self._stages.append(stage)
        return self

    def add_stages(self, stages: List[Stage]) -> "PipelineBuilder":
        """
        Add multiple stages to the pipeline.

        Args:
            stages: List of stage instances to add

        Returns:
            Self for method chaining
        """
        self._stages.extend(stages)
        return self

    def build(self) -> Pipeline:
        """
        Build the pipeline from added stages.

        Returns:
            Constructed Pipeline instance

        Raises:
            ValueError: If no stages were added
        """
        if not self._stages:
            raise ValueError("Cannot build pipeline with no stages")

        return Pipeline(self._stages)

    def reset(self) -> "PipelineBuilder":
        """
        Clear all stages from the builder.

        Returns:
            Self for method chaining
        """
        self._stages.clear()
        return self


def build_default_pipeline(
    *, demosaic_type: str = "EA", apply_orientation: bool = True
) -> Pipeline:
    """
    Build the default ISP pipeline.

    This creates a standard pipeline with all typical ISP stages:
    raw -> normalize -> lens shading -> white balance -> demosaic ->
    XYZ -> sRGB -> gamma -> tone

    Args:
        demosaic_type: Demosaic algorithm type ("EA", "MHC", etc.)
        apply_orientation: Whether to apply EXIF orientation

    Returns:
        Configured Pipeline instance
    """
    stages = [
        NormalizeStage(input_stage="raw", output_stage="normal"),
        LensShadingStage(input_stage="normal", output_stage="lens_shading_correction"),
        WhiteBalanceStage(
            input_stage="lens_shading_correction", output_stage="white_balance"
        ),
        DemosaicStage(
            input_stage="white_balance",
            output_stage="demosaic",
            config={"demosaic_type": demosaic_type, "output_channel_order": "RGB"},
        ),
        XYZStage(input_stage="demosaic", output_stage="xyz"),
        SRGBStage(
            input_stage="xyz",
            output_stage="srgb",
            config={"apply_orientation": apply_orientation},
        ),
        RSBlurCRFStage(input_stage="srgb", output_stage="tone"),
    ]

    return Pipeline(stages)


def build_minimal_pipeline() -> Pipeline:
    """
    Build a minimal pipeline with only essential stages.

    Pipeline: raw -> normalize -> white balance -> demosaic

    Returns:
        Minimal Pipeline instance
    """
    stages = [
        NormalizeStage(input_stage="raw", output_stage="normal"),
        WhiteBalanceStage(
            input_stage="normal",
            output_stage="white_balance",
            config={},  # Will need to handle no lens shading case
        ),
        DemosaicStage(
            input_stage="white_balance",
            output_stage="demosaic",
            config={"demosaic_type": "EA"},
        ),
    ]

    return Pipeline(stages)


def build_custom_pipeline(stage_configs: List[dict]) -> Pipeline:
    """
    Build a custom pipeline from stage configuration dicts.

    Args:
        stage_configs: List of dicts with keys:
            - type: Stage class name or type
            - input_stage: Input stage name
            - output_stage: Output stage name
            - config: Optional stage configuration

    Returns:
        Custom Pipeline instance

    Example:
        configs = [
            {"type": "normalize", "input_stage": "raw", "output_stage": "normal"},
            {"type": "demosaic", "input_stage": "normal", "output_stage": "rgb",
             "config": {"demosaic_type": "MHC"}},
        ]
        pipeline = build_custom_pipeline(configs)
    """
    stage_map = {
        "normalize": NormalizeStage,
        "lens_shading": LensShadingStage,
        "white_balance": WhiteBalanceStage,
        "demosaic": DemosaicStage,
        "xyz": XYZStage,
        "srgb": SRGBStage,
    }

    stages = []
    for cfg in stage_configs:
        stage_type = cfg["type"]

        if isinstance(stage_type, str):
            if stage_type not in stage_map:
                raise ValueError(f"Unknown stage type: {stage_type}")
            stage_class = stage_map[stage_type]
        else:
            stage_class = stage_type

        stage = stage_class(
            input_stage=cfg["input_stage"],
            output_stage=cfg["output_stage"],
            config=cfg.get("config"),
        )
        stages.append(stage)

    return Pipeline(stages)
