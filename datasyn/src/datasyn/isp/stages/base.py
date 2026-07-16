"""
Base classes for stage implementations.

This module provides helper base classes that implement common patterns
for ISP pipeline stages.
"""

from typing import Any, List, Mapping

from numpy.typing import NDArray

from ..core import Stage
from ..metadata import validate_required_keys


class ValidatedStage(Stage):
    """
    Stage with automatic metadata validation.

    Subclasses should define REQUIRED_METADATA_KEYS class attribute.
    """

    REQUIRED_METADATA_KEYS: List[str] = []

    def validate_metadata(self, metadata: Mapping[str, Any]) -> None:
        """
        Validate required metadata keys are present.

        Args:
            metadata: Image metadata to validate

        Raises:
            ValueError: If required keys are missing
        """
        if self.REQUIRED_METADATA_KEYS:
            validate_required_keys(
                metadata,
                self.REQUIRED_METADATA_KEYS,
                stage_name=self.__class__.__name__,
            )


class ConditionalStage(ValidatedStage):
    """
    Stage that can be conditionally skipped based on metadata.

    Subclasses should implement should_apply() method to determine
    whether the stage should be applied.
    """

    def should_apply(self, metadata: Mapping[str, Any]) -> bool:
        """
        Check if stage should be applied based on metadata.

        Args:
            metadata: Image metadata

        Returns:
            True if stage should be applied, False to skip
        """
        return True

    def forward(
        self, image: NDArray, metadata: Mapping[str, Any], config: Mapping[str, Any]
    ) -> NDArray:
        """
        Apply forward transformation conditionally.

        Args:
            image: Input image array
            metadata: Image metadata
            config: Runtime configuration

        Returns:
            Transformed image (or original if skipped)
        """
        if not self.should_apply(metadata):
            return image
        return self.forward_impl(image, metadata, config)

    def inverse(
        self, image: NDArray, metadata: Mapping[str, Any], config: Mapping[str, Any]
    ) -> NDArray:
        """
        Apply inverse transformation conditionally.

        Args:
            image: Input image array
            metadata: Image metadata
            config: Runtime configuration

        Returns:
            Inverse-transformed image (or original if skipped)
        """
        if not self.should_apply(metadata):
            return image
        return self.inverse_impl(image, metadata, config)

    def forward_impl(
        self, image: NDArray, metadata: Mapping[str, Any], config: Mapping[str, Any]
    ) -> NDArray:
        """
        Actual forward implementation (called when should_apply returns True).

        Args:
            image: Input image array
            metadata: Image metadata
            config: Runtime configuration

        Returns:
            Transformed image
        """
        raise NotImplementedError()

    def inverse_impl(
        self, image: NDArray, metadata: Mapping[str, Any], config: Mapping[str, Any]
    ) -> NDArray:
        """
        Actual inverse implementation (called when should_apply returns True).

        Args:
            image: Input image array
            metadata: Image metadata
            config: Runtime configuration

        Returns:
            Inverse-transformed image
        """
        raise NotImplementedError(
            f"Inverse operation not implemented for {self.__class__.__name__}"
        )
