"""
Geometrically augmented patch sampling within a binary validity mask.

Pipeline
--------
1. build_integral_image(mask)          — O(H·W), once per mask.
2. sample_augmented_patch(...)         — O(H·W) per sample, fully JAX.
   a. Compute integer AABB half-extents (e_r, e_c) from L_inv.
   b. erode_and_sample_center: erode mask by that AABB via integral image,
      then sample a center uniformly using the Gumbel-max trick.
   c. center_to_coords + transform_from_coords: inverse-warp the patch.

JAX design notes
----------------
- Erosion is four array slices on the integral image — one vectorised op over
  the full (H·W) spatial domain, no Python loop.
- Uniform sampling from the valid set uses the Gumbel-max trick:
    argmax_i  Gumbel(0,1)_i  s.t. valid[i]
  This is equivalent to uniform sampling and is fully JIT-compatible (no
  dynamic shapes, no host-side sorting or indexing).
- erode_and_sample_center is JIT'd with (e_r, e_c) as static args, so it
  recompiles only on new integer AABB pairs — a small bounded set for any
  finite augmentation range; subsequent calls hit the XLA cache.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
import jax.random as jr

from datasyn.jaxutils import jx
from datasyn.jaxutils.typing import *
from datasyn.synthesis.utils.geoaug_sample import (
    affine_aabb_extents,
    center_to_coords_jit,
    transform_from_coords_jit,
)

# ---------------------------------------------------------------------------
# Integral image  (precomputed once per mask)
# ---------------------------------------------------------------------------


def build_integral_image(mask: JArray) -> JArray:
    """
    Build the 2D prefix-sum (integral image) of a boolean mask.

    Parameters
    ----------
    mask : (H, W) bool or int — True = valid pixel.

    Returns
    -------
    I : (H+1, W+1) int32 — I[i, j] = sum(mask[:i, :j]).
        Zero-padded first row and column for boundary-free lookups.
    """
    I = jnp.cumsum(jnp.cumsum(mask.astype(jnp.int32), axis=0), axis=1)
    return jnp.pad(I, ((1, 0), (1, 0)))


build_integral_image_jit = jx.jit(build_integral_image)


# ---------------------------------------------------------------------------
# Per-sample: erosion + uniform center sampling
# ---------------------------------------------------------------------------


def erode_and_sample_center(
    I: JArray,
    e_r: int,
    e_c: int,
    key: JArray,
) -> tuple[JArray, JArray]:
    """
    Erode the mask by an (e_r, e_c) half-extent rectangle and sample a center
    uniformly from the valid positions using the Gumbel-max trick.

    Parameters
    ----------
    I   : (H+1, W+1) int32 — integral image from build_integral_image.
    e_r : int (static) — row half-extent of the AABB.
    e_c : int (static) — col half-extent of the AABB.
    key : JAX PRNGKey.

    Returns
    -------
    center : (2,) float32 — (row, col) pixel-index coordinates of the center.
    found  : bool scalar — False if no valid center exists (mask too sparse).

    Notes
    -----
    e_r and e_c are static JIT args: recompilation occurs only when a new
    integer pair is first encountered.  The output shape of the erosion kernel
    is (H - 2*e_r, W - 2*e_c) and is fixed at compile time.
    """
    H = I.shape[0] - 1
    W = I.shape[1] - 1
    n_r = H - 2 * e_r
    n_c = W - 2 * e_c

    # Handle degenerate case: AABB too large for image.
    if n_r <= 0 or n_c <= 0:
        return jnp.zeros(2, dtype=jnp.float32), jnp.bool_(False)

    # Erosion via integral image — four static slices, O(n_r·n_c) vectorised.
    #
    # For center (r, c) in [0, n_r) × [0, n_c), the corresponding source pixel
    # is (r + e_r, c + e_c).  The AABB window covers:
    #   rows [r, r + 2*e_r],  cols [c, c + 2*e_c]  (both inclusive).
    # Sum = I[r+2e_r+1, c+2e_c+1] - I[r, c+2e_c+1] - I[r+2e_r+1, c] + I[r, c].
    window_sum = (
        I[2 * e_r + 1 : H + 1, 2 * e_c + 1 : W + 1]
        - I[0:n_r, 2 * e_c + 1 : W + 1]
        - I[2 * e_r + 1 : H + 1, 0:n_c]
        + I[0:n_r, 0:n_c]
    )  # shape (n_r, n_c)
    valid = window_sum == (2 * e_r + 1) * (2 * e_c + 1)
    found = jnp.any(valid)

    # Gumbel-max: assign i.i.d. Gumbel noise, mask invalid positions to -inf,
    # take argmax -> exactly uniform over valid positions.
    gumbel = jr.gumbel(key, shape=(n_r, n_c))
    flat_idx = jnp.argmax(jnp.where(valid, gumbel, -jnp.inf))
    r_local, c_local = jnp.divmod(flat_idx, n_c)

    # Convert erosion-local coordinates back to original image pixel indices.
    center = jnp.stack(
        [
            (r_local + e_r).astype(jnp.float32),
            (c_local + e_c).astype(jnp.float32),
        ]
    )
    return center, found


erode_and_sample_center_jit = jx.jit(
    erode_and_sample_center,
    static_argnames=("e_r", "e_c"),
)


# ---------------------------------------------------------------------------
# Top-level: one augmented patch
# ---------------------------------------------------------------------------


def sample_augmented_patch(
    image: JArray,
    integral_image: JArray,
    patch_hw: Tuple[int, int],
    L_inv: JArray,
    key: JArray,
) -> tuple[JArray, JArray]:
    """
    Sample one geometrically augmented patch from the valid region of the mask.

    Parameters
    ----------
    image          : (H, W) or (H, W, C) source image (JAX array).
    integral_image : (H+1, W+1) int32 — from build_integral_image(mask).
    patch_hw       : (h, w) output patch size (static).
    L_inv          : (2, 2) float — inverse linear map (output offsets -> source offsets).
                     Sampled externally per patch; determines both the AABB used for
                     center validation and the actual inverse warp.
    key            : JAX PRNGKey (consumed).

    Returns
    -------
    patch  : (h, w) or (h, w, C) float array.
    center : (2,) float32 — (row, col) source-image coordinates of the patch center.
    found  : bool scalar — False if no valid center exists; patch and center are
             garbage in that case.

    Notes
    -----
    Computing (e_r, e_c) as Python ints from L_inv requires a small
    device->host sync.  For CPU-resident L_inv (the common case when the
    augmentation transform is sampled on the host) this is free.

    erode_and_sample_center_jit recompiles only on new (e_r, e_c) integer pairs.
    For a bounded augmentation range the full compilation set is reached quickly,
    after which every call hits the XLA cache.
    """
    # Integer AABB half-extents — determines which JIT variant to dispatch.
    e_row, e_col = affine_aabb_extents(patch_hw, L_inv)
    e_r = int(jnp.ceil(float(e_row)))
    e_c = int(jnp.ceil(float(e_col)))

    center, found = erode_and_sample_center_jit(integral_image, e_r, e_c, key)

    coords = center_to_coords_jit(patch_hw, L_inv, center)
    patch = transform_from_coords_jit(image, coords)

    return patch, center, found
