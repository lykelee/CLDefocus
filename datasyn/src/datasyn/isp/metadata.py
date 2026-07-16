"""
Metadata schemas and validation utilities.

This module provides protocols and helper functions for working with image metadata
from DNG/RAW files.
"""

from typing import Any, List, Mapping, Protocol, Union

import numpy as np


class Ratio:
    """Placeholder for Ratio type."""

    pass


# Type aliases for common metadata types
MetadataValue = Union[int, float, str, List, np.ndarray, Ratio]


class MetadataProtocol(Protocol):
    """Protocol defining common metadata structure."""

    def get(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key."""
        ...

    def __getitem__(self, key: str) -> Any:
        """Get metadata value by key."""
        ...

    def __contains__(self, key: str) -> bool:
        """Check if key exists in metadata."""
        ...


def validate_required_keys(
    metadata: Mapping[str, Any], required_keys: List[str], stage_name: str = "Stage"
) -> None:
    """
    Validate that required metadata keys are present.

    Args:
        metadata: Metadata dictionary to validate
        required_keys: List of required key names
        stage_name: Name of the stage for error messages

    Raises:
        ValueError: If any required key is missing
    """
    missing_keys = [key for key in required_keys if key not in metadata]
    if missing_keys:
        raise ValueError(f"{stage_name} requires metadata keys: {missing_keys}")


def normalize_metadata_value(
    value: Any, expected_type: type, allow_list: bool = True
) -> Any:
    """
    Normalize metadata values to expected types.

    Handles common conversions like:
    - Single-element lists to scalars
    - Ratio objects to floats
    - Type coercion

    Args:
        value: Input value to normalize
        expected_type: Expected output type (int, float, etc.)
        allow_list: Whether to allow list types

    Returns:
        Normalized value
    """
    # To avoid circular dependency
    from datasyn.isp import pipeline_utils as pu

    if isinstance(value, pu.Ratio):
        value = float(value)

    if isinstance(value, list) and len(value) == 1 and not allow_list:
        value = value[0]

    if isinstance(value, list) and value and isinstance(value[0], pu.Ratio):
        value = pu.ratios2floats(value)

    if expected_type is not type(value):
        try:
            value = expected_type(value)
        except (TypeError, ValueError):
            pass  # Return as-is if conversion fails

    return value


def get_black_level(metadata: Mapping[str, Any]) -> Union[float, List[float]]:
    """
    Extract and normalize black level from metadata.

    Args:
        metadata: Image metadata

    Returns:
        Black level as float or list of 4 floats (for Bayer pattern)
    """
    from datasyn.isp import pipeline_utils as pu

    black_level = metadata["black_level"]

    if isinstance(black_level, (int, float)):
        return float(black_level)

    if isinstance(black_level, list) and len(black_level) == 1:
        return float(black_level[0])

    # Handle 4-element list (Bayer pattern)
    if isinstance(black_level, list) and len(black_level) == 4:
        if isinstance(black_level[0], pu.Ratio):
            return pu.ratios2floats(black_level)
        return [float(x) for x in black_level]

    return black_level


def get_white_level(metadata: Mapping[str, Any]) -> float:
    """
    Extract and normalize white level from metadata.

    Args:
        metadata: Image metadata

    Returns:
        White level as float
    """
    white_level = metadata["white_level"]

    if isinstance(white_level, list) and len(white_level) == 1:
        return float(white_level[0])

    return float(white_level)


def get_color_matrix(
    metadata: Mapping[str, Any], matrix_key: str = "color_matrix_1"
) -> np.ndarray:
    """
    Extract and normalize color matrix from metadata.

    Args:
        metadata: Image metadata
        matrix_key: Key for color matrix (color_matrix_1 or color_matrix_2)

    Returns:
        3x3 numpy array
    """
    from datasyn.isp import pipeline_utils as pu

    matrix = metadata[matrix_key]

    if isinstance(matrix[0], pu.Ratio):
        matrix = pu.ratios2floats(matrix)

    # Reshape to 3x3
    return np.reshape(np.asarray(matrix), (3, 3))


def get_as_shot_neutral(metadata: Mapping[str, Any]) -> List[float]:
    """
    Extract and normalize as-shot neutral values from metadata.

    Args:
        metadata: Image metadata

    Returns:
        List of 3 floats [R, G, B]
    """
    from datasyn.isp import pipeline_utils as pu

    as_shot_neutral = metadata["as_shot_neutral"]

    if isinstance(as_shot_neutral[0], pu.Ratio):
        return pu.ratios2floats(as_shot_neutral)

    return [float(x) for x in as_shot_neutral]


def get_cfa_pattern(metadata: Mapping[str, Any]) -> List[int]:
    """
    Extract CFA (Color Filter Array) pattern from metadata.

    Args:
        metadata: Image metadata

    Returns:
        List of 4 integers representing Bayer pattern indices
    """
    return metadata["cfa_pattern"]


def get_orientation(metadata: Mapping[str, Any]) -> int:
    """
    Extract image orientation from metadata.

    Args:
        metadata: Image metadata

    Returns:
        Orientation value (EXIF orientation tag)
    """
    orientation = metadata.get("orientation", 1)

    if isinstance(orientation, list):
        return int(orientation[0])

    return int(orientation)
