"""
Core pipeline framework.

This module provides the base Pipeline and Stage classes that form the foundation
of the ISP processing pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional, Sequence

from numpy.typing import NDArray


class Stage(ABC):
    """
    Abstract base class for pipeline stages.

    Each stage represents a single processing step in the ISP pipeline,
    with forward (raw->processed) and optional inverse (processed->raw) operations.
    """

    def __init__(
        self,
        input_stage: str,
        output_stage: str,
        config: Optional[Mapping[str, Any]] = None,
    ):
        """
        Initialize a stage.

        Args:
            input_stage: Name of the input stage state
            output_stage: Name of the output stage state
            config: Optional configuration dictionary for this stage
        """
        self.input_stage = input_stage
        self.output_stage = output_stage
        self.config = config or {}

    @abstractmethod
    def forward(
        self, image: NDArray, metadata: Mapping[str, Any], config: Mapping[str, Any]
    ) -> NDArray:
        """
        Apply forward transformation.

        Args:
            image: Input image array
            metadata: Image metadata (EXIF, DNG tags, etc.)
            config: Runtime configuration overrides

        Returns:
            Transformed image array
        """
        pass

    def inverse(
        self, image: NDArray, metadata: Mapping[str, Any], config: Mapping[str, Any]
    ) -> NDArray:
        """
        Apply inverse transformation.

        Args:
            image: Input image array
            metadata: Image metadata
            config: Runtime configuration overrides

        Returns:
            Inverse-transformed image array

        Raises:
            NotImplementedError: If inverse is not supported for this stage
        """
        raise NotImplementedError(
            f"Inverse operation not implemented for {self.__class__.__name__}"
        )

    def validate_metadata(self, metadata: Mapping[str, Any]) -> None:
        """
        Validate that required metadata keys are present.

        Args:
            metadata: Image metadata to validate

        Raises:
            ValueError: If required metadata is missing
        """
        pass  # Override in subclasses if validation needed

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.input_stage} -> {self.output_stage})"


class Pipeline:
    """Image processing pipeline composed of sequential stages."""

    def __init__(self, stages: Sequence[Stage]):
        """
        Initialize pipeline with stages.

        Args:
            stages: Sequence of Stage objects to execute

        Raises:
            ValueError: If pipeline is empty or stages are not contiguous
        """
        if not stages:
            raise ValueError("Pipeline must contain at least one stage.")

        self.stages = tuple(stages)
        self.stage_order = [stages[0].input_stage] + [s.output_stage for s in stages]

        # Validate unique stage names
        if len(set(self.stage_order)) != len(self.stage_order):
            raise ValueError("Stage names must be unique along the pipeline.")

        # Validate stages are contiguous
        for idx in range(len(stages) - 1):
            if stages[idx].output_stage != stages[idx + 1].input_stage:
                raise ValueError(
                    f"Stages must be contiguous; gap between "
                    f"{stages[idx].output_stage} and {stages[idx + 1].input_stage}"
                )

    def _validate_bounds(self, input_stage: str, output_stage: str) -> None:
        """
        Validate that stage names exist in pipeline.

        Args:
            input_stage: Input stage name
            output_stage: Output stage name

        Raises:
            ValueError: If stage names are not in pipeline
        """
        if input_stage not in self.stage_order:
            raise ValueError(f"Unknown input_stage: {input_stage}")
        if output_stage not in self.stage_order:
            raise ValueError(f"Unknown output_stage: {output_stage}")

    def run(
        self,
        image: NDArray,
        metadata: Mapping[str, Any],
        *,
        input_stage: str,
        output_stage: str,
        direction: str = "forward",
        config_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
        validate: bool = True,
    ) -> NDArray:
        """
        Run pipeline between two stages.

        Args:
            image: Input image array
            metadata: Image metadata dictionary
            input_stage: Starting stage name
            output_stage: Ending stage name
            direction: Either "forward" or "inverse"
            config_overrides: Optional per-stage config overrides
            validate: Whether to validate metadata before running

        Returns:
            Processed image array

        Raises:
            ValueError: If direction is invalid or stage order is wrong
        """
        direction = direction.lower()
        if direction not in ("forward", "inverse"):
            raise ValueError(
                f"direction must be 'forward' or 'inverse', got {direction}"
            )

        self._validate_bounds(input_stage, output_stage)
        start_idx = self.stage_order.index(input_stage)
        end_idx = self.stage_order.index(output_stage)

        # Validate direction matches stage ordering
        if direction == "forward" and start_idx > end_idx:
            raise ValueError(
                f"Invalid stage order for forward: {input_stage} occurs after {output_stage}"
            )
        if direction == "inverse" and start_idx < end_idx:
            raise ValueError(
                f"Invalid stage order for inverse: {input_stage} occurs before {output_stage}"
            )

        # Early return if same stage
        if input_stage == output_stage:
            return image

        overrides = config_overrides or {}

        # Run forward direction
        if direction == "forward":
            for idx in range(start_idx, end_idx):
                stage = self.stages[idx]

                # Validate metadata if requested
                if validate:
                    stage.validate_metadata(metadata)

                # Merge config
                cfg = {**stage.config, **overrides.get(stage.output_stage, {})}

                # Execute stage
                image = stage.forward(image, metadata, cfg)

            return image

        # Run inverse direction
        for idx in range(start_idx - 1, end_idx - 1, -1):
            stage = self.stages[idx]

            # Validate metadata if requested
            if validate:
                stage.validate_metadata(metadata)

            # Merge config
            cfg = {**stage.config, **overrides.get(stage.output_stage, {})}

            # Execute inverse
            image = stage.inverse(image, metadata, cfg)

        return image

    def get_stage_by_name(self, output_stage: str) -> Optional[Stage]:
        """
        Get stage by its output stage name.

        Args:
            output_stage: Output stage name to search for

        Returns:
            Stage object or None if not found
        """
        for stage in self.stages:
            if stage.output_stage == output_stage:
                return stage
        return None

    def __repr__(self) -> str:
        stage_names = " -> ".join(self.stage_order)
        return f"Pipeline({stage_names})"
