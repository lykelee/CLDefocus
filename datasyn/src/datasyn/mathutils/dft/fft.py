"""
FFT operations.

These functions automatically handle offset conversion:
1. Convert input to DFT arrangement (offset=0) if needed
2. Perform FFT
3. Convert output back to the original offset

The relationship between spatial and frequency samplers:
- Spatial:   spacing dx, shape N, extent L = N*dx
- Frequency: spacing df = 1/L = 1/(N*dx), shape N, extent 1/dx
"""

from typing import Callable, Literal

import jax.numpy as jnp

from datasyn.jaxutils.typing import *
from datasyn.mathutils.dft.sampler import DFTSampleGrid, DFTSampler
from datasyn.mathutils.grid import StaticShape

NormType = Literal["backward", "ortho", "forward"]


def _compute_new_spacing(
    grid: DFTSampleGrid,
    axes: tuple[int, ...],
    values_ndim: int,
) -> FloatArray:
    """
    Compute output spacing after a DFT along the given axes.

    For each transformed axis, the spacing becomes df = 1/(N*dx).
    Axes that don't correspond to a sampler dimension are ignored.

    Args:
        grid:       Grid in DFT arrangement (offset=0).
        axes:       Array axes that were transformed (non-negative).
        values_ndim: Total number of dimensions of the values array.

    Returns:
        New spacing array. Shape (ndim,).
    """
    new_spacing = grid.spacing
    sampler_offset = values_ndim - grid.ndim  # leading batch dims
    for ax in axes:
        sampler_axis = ax - sampler_offset
        if 0 <= sampler_axis < grid.ndim:
            new_spacing = new_spacing.at[sampler_axis].set(
                1.0 / (grid.shape[sampler_axis] * grid.spacing[sampler_axis])
            )
    return new_spacing


def _resolve_axes(
    axes: tuple[int, ...] | None, values_ndim: int, ndim: int
) -> tuple[int, ...]:
    """
    Resolve None axes to the sampler dimensions, normalising negative indices.

    Args:
        axes:        User-provided axes, or None for all sampler dimensions.
        values_ndim: Total ndim of the values array.
        ndim:        Number of sampler dimensions.

    Returns:
        Tuple of non-negative axis indices.
    """
    if axes is None:
        return tuple(range(values_ndim - ndim, values_ndim))
    return tuple(ax if ax >= 0 else values_ndim + ax for ax in axes)


def _apply_transform(
    grid: DFTSampleGrid,
    transform_fn: Callable,
    axes: tuple[int, ...] | None,
    norm: NormType,
) -> DFTSampleGrid:
    """
    Core helper: convert to DFT arrangement, apply transform, restore offset.

    Args:
        grid:         Input DFTSampleGrid (any offset).
        transform_fn: jnp.fft.fftn or jnp.fft.ifftn (same signature).
        axes:         Axes to transform. None means all sampler dimensions.
        norm:         Normalization mode.

    Returns:
        Transformed DFTSampleGrid with same offset as input.
    """
    original_offset = grid.offset
    values_ndim = grid.values.ndim

    grid = grid.to_dft_arrangement()
    axes = _resolve_axes(axes, values_ndim, grid.ndim)

    result_values = transform_fn(grid.values, axes=axes, norm=norm)
    new_spacing = _compute_new_spacing(grid, axes, values_ndim)

    new_sampler = DFTSampler(
        spacing=new_spacing,
        shape=grid.shape,
        offset=_zero_offset(grid.ndim),
    )
    result = DFTSampleGrid(sampler=new_sampler, values=result_values)
    return result.with_offset(original_offset)


def _zero_offset(ndim: int) -> tuple[int, ...]:
    return (0,) * ndim


# ---------------------------------------------------------------------------
# N-dimensional transforms
# ---------------------------------------------------------------------------


def fftn(
    grid: DFTSampleGrid,
    axes: tuple[int, ...] | None = None,
    norm: NormType = "backward",
) -> DFTSampleGrid:
    """
    N-dimensional FFT on a DFTSampleGrid.

    Automatically handles offset: converts to DFT arrangement before FFT,
    restores the original offset on output.

    Args:
        grid: Spatial domain samples (any offset arrangement).
        axes: Axes to transform. None transforms all sampler dimensions.
        norm: Normalization mode.

    Returns:
        Frequency domain DFTSampleGrid with same offset as input.
    """
    return _apply_transform(grid, jnp.fft.fftn, axes, norm)


def ifftn(
    grid: DFTSampleGrid,
    axes: tuple[int, ...] | None = None,
    norm: NormType = "backward",
) -> DFTSampleGrid:
    """
    N-dimensional inverse FFT on a DFTSampleGrid.

    Automatically handles offset: converts to DFT arrangement before IFFT,
    restores the original offset on output.

    Args:
        grid: Frequency domain samples (any offset arrangement).
        axes: Axes to transform. None transforms all sampler dimensions.
        norm: Normalization mode.

    Returns:
        Spatial domain DFTSampleGrid with same offset as input.
    """
    return _apply_transform(grid, jnp.fft.ifftn, axes, norm)


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def fft(
    grid: DFTSampleGrid,
    axis: int = -1,
    norm: NormType = "backward",
) -> DFTSampleGrid:
    """1D FFT on a DFTSampleGrid."""
    values_ndim = grid.values.ndim
    axis = axis if axis >= 0 else values_ndim + axis
    return fftn(grid, axes=(axis,), norm=norm)


def ifft(
    grid: DFTSampleGrid,
    axis: int = -1,
    norm: NormType = "backward",
) -> DFTSampleGrid:
    """1D inverse FFT on a DFTSampleGrid."""
    values_ndim = grid.values.ndim
    axis = axis if axis >= 0 else values_ndim + axis
    return ifftn(grid, axes=(axis,), norm=norm)


def fft2(
    grid: DFTSampleGrid,
    axes: tuple[int, int] = (-2, -1),
    norm: NormType = "backward",
) -> DFTSampleGrid:
    """2D FFT on a DFTSampleGrid. Convenience wrapper around fftn."""
    return fftn(grid, axes=axes, norm=norm)


def ifft2(
    grid: DFTSampleGrid,
    axes: tuple[int, int] = (-2, -1),
    norm: NormType = "backward",
) -> DFTSampleGrid:
    """2D inverse FFT on a DFTSampleGrid. Convenience wrapper around ifftn."""
    return ifftn(grid, axes=axes, norm=norm)
