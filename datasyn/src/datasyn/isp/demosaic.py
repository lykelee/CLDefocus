"""
DDFAPD - Menon (2007) Bayer CFA Demosaicing — JAX reimplementation
===================================================================

Pure-JAX reimplementation of the DDFAPD Menon (2007) demosaicing algorithm.
Eliminates colour/scipy dependencies; JIT-compiled end-to-end; vmap-compatible
for batch processing.

Reference: Menon, Daniele, Stefano Andriani, and Giancarlo Calvagno. "Demosaicing with directional filtering and a posteriori decision." IEEE Transactions on Image Processing 16.1 (2006): 132-141.
"""

from __future__ import annotations

import functools
from typing import Literal

import jax
import jax.numpy as jnp

from datasyn.jaxutils import jx
from datasyn.jaxutils.typing import *

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "masks_CFA_Bayer",
    "demosaicing_CFA_Bayer_Menon2007",
    "demosaicing_CFA_Bayer_DDFAPD",
    "refining_step_Menon2007",
]


# ---------------------------------------------------------------------------
# Bayer masks
# ---------------------------------------------------------------------------


def masks_CFA_Bayer(
    shape: tuple[int, int],
    pattern: str = "RGGB",
) -> tuple[JArray, JArray, JArray]:
    """
    Return boolean (R, G, B) Bayer CFA masks for given shape and pattern.

    Parameters
    ----------
    shape
        (H, W) dimensions of the Bayer CFA.
    pattern
        Arrangement of colour filters, one of "RGGB", "BGGR", "GRBG", "GBRG".

    Returns
    -------
    tuple[JArray, JArray, JArray]
        Boolean R, G, B masks of shape (H, W).

    Notes
    -----
    When called inside a ``jax.jit``-compiled function with ``pattern`` listed
    in ``static_argnames``, the masks become compile-time constants and are
    folded away — no runtime overhead.
    """
    pattern = pattern.upper()
    channels: dict[str, JArray] = {
        ch: jnp.zeros(shape, dtype=jnp.bool_) for ch in "RGB"
    }
    for ch, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[ch] = channels[ch].at[y::2, x::2].set(True)
    return channels["R"], channels["G"], channels["B"]


# ---------------------------------------------------------------------------
# Convolution helpers
#
# scipy.ndimage.convolve (2-D, mode="constant") is TRUE convolution —
# the kernel is flipped 180°. jax.lax.conv_general_dilated is cross-
# correlation (no flip), so we flip the kernel once at trace time.
# All 1-D kernels used by this algorithm are symmetric, so no flip is needed
# there.
# ---------------------------------------------------------------------------


def _cnv_h(x: JArray, kernel: JArray) -> JArray:
    """1-D horizontal convolution with reflect (mirror) boundary."""
    p = kernel.shape[0] // 2
    x = jnp.pad(x, ((0, 0), (p, p)), mode="reflect")
    return jax.lax.conv_general_dilated(
        x[None, None],
        kernel[None, None, None, :],  # (O=1, I=1, kH=1, kW)
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )[0, 0]


def _cnv_v(x: JArray, kernel: JArray) -> JArray:
    """1-D vertical convolution with reflect (mirror) boundary."""
    p = kernel.shape[0] // 2
    x = jnp.pad(x, ((p, p), (0, 0)), mode="reflect")
    return jax.lax.conv_general_dilated(
        x[None, None],
        kernel[None, None, :, None],  # (O=1, I=1, kH, kW=1)
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )[0, 0]


def _cnv_2d(x: JArray, kernel: JArray) -> JArray:
    """True 2-D convolution (kernel flipped) with zero (constant) boundary."""
    pH, pW = kernel.shape[0] // 2, kernel.shape[1] // 2
    x = jnp.pad(x, ((pH, pH), (pW, pW)))  # zero-pad
    return jax.lax.conv_general_dilated(
        x[None, None],
        jnp.flip(kernel)[None, None],  # flip -> true convolution, not correlation
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )[0, 0]


# ---------------------------------------------------------------------------
# Refining step
# ---------------------------------------------------------------------------


@jx.jit
def refining_step_Menon2007(
    RGB: JArray,
    RGB_m: JArray,
    M: JArray,
) -> JArray:
    """
    Perform the refining step on a demosaiced RGB array.

    Parameters
    ----------
    RGB
        Demosaiced RGB array, shape (H, W, 3).
    RGB_m
        Boolean CFA masks stacked as (H, W, 3) with channel order R, G, B.
    M
        Directional decision mask from the main demosaicing step; True where
        the horizontal estimate was chosen over the vertical one.

    Returns
    -------
    JArray
        Refined RGB array, shape (H, W, 3).
    """
    dtype = RGB.dtype
    R, G, B = RGB[..., 0], RGB[..., 1], RGB[..., 2]
    R_m = RGB_m[..., 0].astype(jnp.bool_)
    G_m = RGB_m[..., 1].astype(jnp.bool_)
    B_m = RGB_m[..., 2].astype(jnp.bool_)

    # ------------------------------------------------------------------
    # 1. Update green at known R / B sites
    # ------------------------------------------------------------------
    FIR = jnp.ones(3, dtype=dtype) / 3.0

    R_G = R - G
    B_G = B - G

    R_G_m = jnp.where(R_m, jnp.where(M, _cnv_h(R_G, FIR), _cnv_v(R_G, FIR)), 0.0)
    B_G_m = jnp.where(B_m, jnp.where(M, _cnv_h(B_G, FIR), _cnv_v(B_G, FIR)), 0.0)

    G = jnp.where(R_m, R - R_G_m, G)
    G = jnp.where(B_m, B - B_G_m, G)

    # ------------------------------------------------------------------
    # 2. Update R and B at green sites
    # ------------------------------------------------------------------
    # Row / column indicators — shape (H, 1) / (1, W), broadcast freely.
    R_r = jnp.any(R_m, axis=1, keepdims=True)
    R_c = jnp.any(R_m, axis=0, keepdims=True)
    B_r = jnp.any(B_m, axis=1, keepdims=True)
    B_c = jnp.any(B_m, axis=0, keepdims=True)

    k_b = jnp.array([0.5, 0.0, 0.5], dtype=dtype)

    # Recompute colour differences with the updated G.
    R_G = R - G
    B_G = B - G

    R_G_m = jnp.where(G_m & B_r, _cnv_v(R_G, k_b), R_G_m)
    R = jnp.where(G_m & B_r, G + R_G_m, R)
    R_G_m = jnp.where(G_m & B_c, _cnv_h(R_G, k_b), R_G_m)
    R = jnp.where(G_m & B_c, G + R_G_m, R)

    B_G_m = jnp.where(G_m & R_r, _cnv_v(B_G, k_b), B_G_m)
    B = jnp.where(G_m & R_r, G + B_G_m, B)
    B_G_m = jnp.where(G_m & R_c, _cnv_h(B_G, k_b), B_G_m)
    B = jnp.where(G_m & R_c, G + B_G_m, B)

    # ------------------------------------------------------------------
    # 3. Update R at B sites, then B at R sites
    #    R_B is frozen before the R update; the B update uses the new R.
    # ------------------------------------------------------------------
    R_B = R - B

    R_B_m = jnp.where(B_m, jnp.where(M, _cnv_h(R_B, FIR), _cnv_v(R_B, FIR)), 0.0)
    R = jnp.where(B_m, B + R_B_m, R)

    R_B_m = jnp.where(R_m, jnp.where(M, _cnv_h(R_B, FIR), _cnv_v(R_B, FIR)), 0.0)
    B = jnp.where(R_m, R - R_B_m, B)

    return jnp.stack([R, G, B], axis=-1)


# ---------------------------------------------------------------------------
# Main demosaicing function
# ---------------------------------------------------------------------------


@functools.partial(jx.jit, static_argnames=["pattern", "refining_step"])
def demosaicing_CFA_Bayer_Menon2007(
    CFA: JArray,
    pattern: Literal["RGGB", "BGGR", "GRBG", "GBRG"] | str = "RGGB",
    refining_step: bool = True,
) -> JArray:
    """
    Demosaice a Bayer CFA using DDFAPD - Menon (2007).

    Parameters
    ----------
    CFA
        2-D float array of shape (H, W), raw Bayer sensor data.
    pattern
        Bayer pattern, one of "RGGB", "BGGR", "GRBG", "GBRG".
    refining_step
        Whether to apply the iterative refining step.

    Returns
    -------
    JArray
        Demosaiced RGB image, shape (H, W, 3).

    Notes
    -----
    ``pattern`` and ``refining_step`` are declared ``static_argnames`` so that
    JAX recompiles (once) for each distinct combination, turning the Bayer masks
    and the ``if refining_step`` branch into compile-time constants.

    For batch processing use ``jax.vmap``::

        demosaicing_batch = jax.vmap(
            functools.partial(demosaicing_CFA_Bayer_Menon2007, pattern="RGGB")
        )
        RGB = demosaicing_batch(CFA_batch)  # (B, H, W) -> (B, H, W, 3)
    """
    CFA = jnp.asarray(CFA)
    CFA = jnp.squeeze(CFA)
    dtype = CFA.dtype
    R_m, G_m, B_m = masks_CFA_Bayer(CFA.shape, pattern)

    h_0 = jnp.array([0.0, 0.5, 0.0, 0.5, 0.0], dtype=dtype)
    h_1 = jnp.array([-0.25, 0.0, 0.5, 0.0, -0.25], dtype=dtype)

    R = CFA * R_m
    G = CFA * G_m
    B = CFA * B_m

    # ------------------------------------------------------------------
    # 1. Directional green interpolation
    # ------------------------------------------------------------------
    G_H = jnp.where(~G_m, _cnv_h(CFA, h_0) + _cnv_h(CFA, h_1), G)
    G_V = jnp.where(~G_m, _cnv_v(CFA, h_0) + _cnv_v(CFA, h_1), G)

    # Colour differences at known R / B sites for the directional decision
    C_H = jnp.where(R_m, R - G_H, 0.0)
    C_H = jnp.where(B_m, B - G_H, C_H)

    C_V = jnp.where(R_m, R - G_V, 0.0)
    C_V = jnp.where(B_m, B - G_V, C_V)

    # 2-pixel finite differences of the colour difference signals
    D_H = jnp.abs(C_H - jnp.pad(C_H, ((0, 0), (0, 2)), mode="reflect")[:, 2:])
    D_V = jnp.abs(C_V - jnp.pad(C_V, ((0, 2), (0, 0)), mode="reflect")[2:, :])

    # Aggregation kernel — applied as true convolution and its transpose
    k = jnp.array(
        [
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )

    d_H = _cnv_2d(D_H, k)
    d_V = _cnv_2d(D_V, k.T)

    # M=True -> horizontal estimate is better
    M = d_V >= d_H
    G = jnp.where(M, G_H, G_V)

    # ------------------------------------------------------------------
    # 2. R / B interpolation at green sites
    # ------------------------------------------------------------------
    # Row indicators, shape (H, 1) — broadcast against any (H, W) array.
    R_r = jnp.any(R_m, axis=1, keepdims=True)
    B_r = jnp.any(B_m, axis=1, keepdims=True)

    k_b = jnp.array([0.5, 0.0, 0.5], dtype=dtype)

    # R at G sites on R rows (horizontal interpolation)
    R = jnp.where(G_m & R_r, G + _cnv_h(R, k_b) - _cnv_h(G, k_b), R)
    # R at G sites on B rows (vertical interpolation)
    R = jnp.where(G_m & B_r, G + _cnv_v(R, k_b) - _cnv_v(G, k_b), R)

    # B at G sites on B rows (horizontal interpolation)
    B = jnp.where(G_m & B_r, G + _cnv_h(B, k_b) - _cnv_h(G, k_b), B)
    # B at G sites on R rows (vertical interpolation)
    B = jnp.where(G_m & R_r, G + _cnv_v(B, k_b) - _cnv_v(G, k_b), B)

    # ------------------------------------------------------------------
    # 3. R / B interpolation at the opposite colour sites
    # ------------------------------------------------------------------
    R = jnp.where(
        B_r & B_m,
        jnp.where(
            M,
            B + _cnv_h(R, k_b) - _cnv_h(B, k_b),
            B + _cnv_v(R, k_b) - _cnv_v(B, k_b),
        ),
        R,
    )

    B = jnp.where(
        R_r & R_m,
        jnp.where(
            M,
            R + _cnv_h(B, k_b) - _cnv_h(R, k_b),
            R + _cnv_v(B, k_b) - _cnv_v(R, k_b),
        ),
        B,
    )

    RGB = jnp.stack([R, G, B], axis=-1)

    if refining_step:
        RGB = refining_step_Menon2007(RGB, jnp.stack([R_m, G_m, B_m], axis=-1), M)

    return RGB


demosaicing_CFA_Bayer_DDFAPD = demosaicing_CFA_Bayer_Menon2007
