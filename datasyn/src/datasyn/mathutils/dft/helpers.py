"""
Helpers for creating and working with DFTSampleGrid.

With the offset-based design, grids track their physical positions via the
sampler's offset field. FFT functions handle offset conversion internally.

Common workflows:
- sample_centered: Create grid with centered positions (offset = -N//2)
- sample_at_positions: Create grid at any sampler's positions
- Grids know their positions via grid.positions()
"""

from typing import Callable, NamedTuple

import jax.numpy as jnp

from datasyn.jaxutils.typing import *
from datasyn.mathutils.dft.sampler import DFTSampleGrid, DFTSampler


def next_pow2_int(n: int) -> int:
    """Return the smallest power of 2 >= n (n must be positive)."""
    if n <= 1:
        return 1
    return 1 << ((int(n) - 1).bit_length())


def next_pow2_int_jax_fast(n: IntArray) -> IntArray:
    """
    Find least 2^k >= n with no loop using log/power.

    TODO: I haven't checked this has error.
    """
    n = jnp.asarray(n)
    n = jnp.maximum(n, 1)
    exponent = jnp.ceil(jnp.log2(n))
    return jnp.power(2, exponent).astype(n.dtype)


def next_km_int(m: int, n: int) -> int:
    """
    Return the smallest k * m >= n.

    For example, for m = 10, f(10) = f(11) = ... = f(19) = 10 and f(20) = 20.
    """
    return m * ((n + m - 1) // m)


def centered_positions_1d(spacing: float, n: int) -> FloatArray:
    """
    Generate centered positions for one axis.

    Args:
        spacing: Sample spacing.
        n: Number of samples.

    Returns:
        Positions [(0 - n/2)*dx, (1 - n/2)*dx, ..., (n-1 - n/2)*dx].
        For even n=8: [-4, -3, -2, -1, 0, 1, 2, 3] * dx
        For odd n=7: [-3, -2, -1, 0, 1, 2, 3] * dx
    """
    return (jnp.arange(n) - n // 2) * spacing


def centered_positions(
    spacing: FloatArray | Sequence[float],
    shape: tuple[int, ...] | Sequence[int],
) -> FloatArray:
    """
    Generate centered positions as a meshgrid.

    Args:
        spacing: Spacing per axis.
        shape: Number of samples per axis.

    Returns:
        Array of shape (*shape, ndim) containing centered position vectors.
    """
    spacing = jnp.asarray(spacing)
    shape = tuple(shape)
    axes = [
        centered_positions_1d(float(spacing[ax]), shape[ax]) for ax in range(len(shape))
    ]
    grids = jnp.meshgrid(*axes, indexing="ij")
    return jnp.stack(grids, axis=-1)


def sample_centered(
    func: Callable[[FloatArray], JArray],
    spacing: FloatArray | Sequence[float],
    shape: tuple[int, ...] | Sequence[int],
) -> DFTSampleGrid:
    """
    Sample a function at centered positions.

    Creates a DFTSampleGrid with centered offset (offset = -N//2 per axis).
    The grid tracks that samples are at positions (i - N/2) * spacing.

    FFT functions will automatically handle the offset conversion.

    Args:
        func: Function f(x) -> values. Takes position array of shape (..., ndim)
              and returns values of shape (...) or (..., value_dims).
        spacing: Sample spacing per axis.
        shape: Number of samples per axis.

    Returns:
        DFTSampleGrid with centered offset. Call fft2(grid) directly.
    """
    # Create centered sampler
    sampler = DFTSampler.create_centered(spacing=spacing, shape=shape)

    # Evaluate function at centered positions
    pos = sampler.positions()
    values = func(pos)

    return DFTSampleGrid(sampler=sampler, values=values)


def sample_at_positions(
    func: Callable[[FloatArray], JArray],
    sampler: DFTSampler,
) -> DFTSampleGrid:
    """
    Sample a function at a sampler's positions.

    Use this when you have a specific sampler (with any offset) and want
    to evaluate a function at its positions.

    Args:
        func: Function f(x) -> values.
        sampler: The DFT sampler defining positions and offset.

    Returns:
        DFTSampleGrid with values at sampler's positions.
    """
    pos = sampler.positions()
    values = func(pos)
    return DFTSampleGrid(sampler=sampler, values=values)


class CenteredView(NamedTuple):
    """Result of to_centered_view: centered positions and values."""

    positions: FloatArray
    """Centered positions: (i - N/2) * spacing. Shape (*shape, ndim)."""

    values: JArray
    """Values at centered positions. Shape (*shape, ...)."""


def to_centered_view(grid: DFTSampleGrid) -> CenteredView:
    """
    Get a centered view of a grid for visualization/analysis.

    If the grid is not already centered, this converts it to centered
    arrangement and returns the positions and values.

    Args:
        grid: DFTSampleGrid in any arrangement.

    Returns:
        CenteredView with:
        - positions: Centered position array, shape (*shape, ndim)
        - values: Values at centered positions
    """
    centered_grid = grid.to_centered()
    return CenteredView(
        positions=centered_grid.positions(),
        values=centered_grid.values,
    )


def from_centered_values(
    values: JArray,
    spacing: FloatArray | Sequence[float],
    shape: tuple[int, ...] | Sequence[int],
) -> DFTSampleGrid:
    """
    Create DFTSampleGrid from values at centered positions.

    Creates a grid with centered offset tracking that the values
    are at positions (i - N/2) * spacing.

    Args:
        values: Values at centered positions.
        spacing: Sample spacing per axis.
        shape: Number of samples per axis.

    Returns:
        DFTSampleGrid with centered offset.
    """
    sampler = DFTSampler.create_centered(spacing=spacing, shape=shape)
    return DFTSampleGrid(sampler=sampler, values=values)
