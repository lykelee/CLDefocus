"""
Bluestein's algorithm for scaled & zoomed FFT.

Following terminologies of Zhang, Wenhui, et al., we define "scaling" and "zooming" as:

- scaling: Ability to adjust sampling pitches (=> pixel size).
- zooming: Ability to adjust a region of interest (ROI).

Unlike standard FFT, Bluestein's algorithm using CZT can achieve both.

References: Zhang, Wenhui, et al. "Analysis of numerical diffraction calculation methods: from the perspective of phase space optics and the sampling theorem." Journal of the Optical Society of America A 37.11 (2020): 1748-1766.
"""

import jax.numpy as jnp

from datasyn.jaxutils.nputils import slice_along_axis
from datasyn.jaxutils.typing import *
from datasyn.mathutils.dft.helpers import next_pow2_int
from datasyn.mathutils.dft.sampler import DFTSampleGrid
from datasyn.mathutils.grid import BoundedGrid


def _mul_along_axis(a: JArray, b: JArray, axis: int):
    axis = axis % a.ndim
    if b.ndim != 1:
        raise ValueError(f"b must be 1D, got shape {b.shape}")
    if b.shape[0] != a.shape[axis]:
        raise ValueError(
            f"b has length {b.shape[0]} but a.shape[{axis}] is {a.shape[axis]}"
        )
    shape = (1,) * axis + (b.shape[0],) + (1,) * (a.ndim - axis - 1)
    return a * b.reshape(shape)


def bluestein_dft(
    grid: DFTSampleGrid,
    f1: float,
    f2: float,
    mout: int,
    axis: int = -1,
) -> tuple[JArray, BoundedGrid]:
    axis = axis % grid.ndim
    grid = grid.to_centered(axis=axis)

    x = grid.values
    m = x.shape[axis]
    mp = m + mout - 1
    nfft = next_pow2_int(mp)

    dtype_r = x.real.dtype
    tau = 2.0 * jnp.pi

    fs = 1.0 / grid.spacing[axis]

    df = f2 - f1
    f11 = f1 + df / (2 * mout)
    f22 = f2 + df / (2 * mout)

    a = jnp.exp(1j * tau * f11 / fs)
    w = jnp.exp(-1j * tau * (f22 - f11) / (mout * fs))

    h_idx_end = max(mout - 1, m - 1)
    h_idx = jnp.arange(-m + 1, h_idx_end + 1, dtype=jnp.int32)
    h_idx_r = h_idx.astype(dtype_r)

    h = w ** (0.5 * (h_idx_r**2))

    Fh = jnp.fft.fft(1.0 / h[:mp], n=nfft)

    n = jnp.arange(m, dtype=dtype_r)
    b_chirp = (a ** (-n)) * h[m - 1 : 2 * m - 1]

    tt = jnp.fft.fft(_mul_along_axis(x, b_chirp, axis=axis), n=nfft, axis=axis)

    y = jnp.fft.ifft(_mul_along_axis(tt, Fh, axis=axis), n=nfft, axis=axis)

    y = slice_along_axis(y, slice(m - 1, mp), axis=axis)
    y = _mul_along_axis(y, h[m - 1 : mp], axis=axis)

    k = jnp.arange(mout, dtype=dtype_r)
    l = (k / mout) * (f22 - f11) + f11

    phase = jnp.exp(-1j * tau * l * (-m / 2.0 + 0.5) / fs)
    y = _mul_along_axis(y, phase, axis=axis)

    # Physical output grid you probably want to expose:
    # the actual visible band is still the interval [f1, f2], sampled at bin centers
    dy = (f2 - f1) / mout
    y_grid = BoundedGrid.from_origin_step(
        origin=jnp.array([f1 + dy / 2], dtype=dtype_r),
        step=jnp.array([dy], dtype=dtype_r),
        shape=(mout,),
    )

    return y, y_grid


def bluestein_dft_nd(
    grid: DFTSampleGrid,
    f1: FloatArray,
    f2: FloatArray,
    mout: int | tuple[int, ...],
    axis: int | tuple[int, ...],
) -> tuple[JArray, BoundedGrid]:
    if isinstance(axis, int):
        axis = (axis,)
    y_grids = []
    for i, a in enumerate(axis):
        y, y_grid = bluestein_dft(grid, f1=f1[i], f2=f2[i], mout=mout[i], axis=a)
        grid = grid.with_values(y)
        y_grids.append(y_grid)
    y_grid = BoundedGrid.stack(y_grids)
    return grid.values, y_grid


def bluestein_dft_2d(
    grid: DFTSampleGrid,
    f1: FloatArray,
    f2: FloatArray,
    mout: tuple[int, int],
    axis: tuple[int, int] = (-2, -1),
) -> tuple[JArray, BoundedGrid]:
    return bluestein_dft_nd(grid=grid, f1=f1, f2=f2, mout=mout, axis=axis)
