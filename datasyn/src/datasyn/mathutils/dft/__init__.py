"""
DFT (Discrete Fourier Transform) utilities with explicit grid management.

This module provides FFT operations that track the relationship between
spatial and frequency domain sampling. All operations use the native DFT
arrangement (index i -> position i * spacing), with explicit helpers for
converting to/from centered arrangements.

Key classes:
    DFTSampler: Sampling specification (spacing + shape)
    DFTSampleGrid: Sampler + value array

Helpers for centered arrangement:
    sample_centered: Sample function at centered positions, return DFT arrangement
    to_centered_view: Convert DFT arrangement to centered view for visualization
    from_centered_values: Convert centered values to DFT arrangement
"""

from datasyn.mathutils.dft.fft import (
    fft,
    fft2,
    fftn,
    ifft,
    ifft2,
    ifftn,
)
from datasyn.mathutils.dft.helpers import (
    CenteredView,
    centered_positions,
    centered_positions_1d,
    from_centered_values,
    sample_at_positions,
    sample_centered,
    to_centered_view,
)
from datasyn.mathutils.dft.sampler import DFTSampleGrid, DFTSampler

__all__ = [
    # Core classes
    "DFTSampler",
    "DFTSampleGrid",
    # FFT functions
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    # Helpers
    "sample_centered",
    "sample_at_positions",
    "to_centered_view",
    "from_centered_values",
    "centered_positions",
    "centered_positions_1d",
    "CenteredView",
]
