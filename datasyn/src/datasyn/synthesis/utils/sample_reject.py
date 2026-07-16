"""
Rejection-sampling of geometrically augmented patches within a binary mask.

Four-step pipeline
------------------
1. preprocess_mask(mask)
       O(H·W), once per mask.
       Returns a MaskData object (opaque — holds the integral image internally).

2. sample_valid_centers(mask_data, patch_hw, L_invs, key)
       O(N) — center sampling + batched AABB extents + vectorised validity check.
       Returns a SampledCenters object with M ≤ N accepted (center, L_inv) pairs.
       No image needed.

3. prepare_coords(patch_hw, centers, L_invs)
       O(M·h·w), image-independent.
       Batched inverse-warp grid computation for all M patches in one JIT'd call.
       Returns a PreparedCoords object reusable across all co-registered images.

4. extract_patches(image, prepared_coords)
       O(M·h·w) — apply precomputed coords to one image.
       Call once per image; coords are not recomputed.

Data flow for one mask / multiple images
-----------------------------------------
    mask_data = preprocess_mask(mask)
    sampled   = sample_valid_centers(mask_data, patch_hw, L_invs, key)
    prepared  = prepare_coords(patch_hw, sampled.centers, sampled.L_invs)  # once

    patches_a = extract_patches(image_a, prepared)
    patches_b = extract_patches(image_b, prepared)

Comparison with sample_geoaug.py
---------------------------------
sample_geoaug  :  erodes the full (H·W) mask for a fixed L_inv, then draws one
                  guaranteed-valid center.  Scales as O(N·H·W) when L_inv varies.

sample_reject  :  4 integral-image lookups per candidate -> O(N) check; warp only
                  accepted patches.  Correct choice when L_inv varies per sample.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from datasyn.jaxutils import jx
from datasyn.jaxutils.typing import *
from datasyn.synthesis.utils.geoaug_sample import (
    center_to_coords,
    transform_from_coords,
)
from datasyn.synthesis.utils.sample_geoaug import build_integral_image_jit

# ---------------------------------------------------------------------------
# Data objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaskData:
    """
    Preprocessed mask for rejection-sampling patch centers.

    Construct via preprocess_mask().  Treat as opaque — the only public field
    intended for direct use is `shape`.
    """

    shape: Tuple[int, int]  # (H, W)
    _integral: JArray = field(repr=False)  # (H+1, W+1) int32, internal


@dataclass(frozen=True)
class SampledCenters:
    """
    Accepted (center, L_inv) pairs from sample_valid_centers.

    Pass to prepare_coords() to compute source-space coordinates.
    `valid_mask` carries the per-candidate acceptance flags for the original N
    candidates — useful for tracking acceptance rates or analysing the distribution.
    """

    centers: JArray  # (M, 2)  float32 — (row, col) of accepted centers
    L_invs: JArray  # (M, 2, 2) float — corresponding inverse maps
    valid_mask: JArray  # (N,)  bool  — acceptance flag for all N input candidates


@dataclass(frozen=True)
class PreparedCoords:
    """
    Precomputed source-space sampling coordinates for M patches.

    Computed once from (centers, L_invs) by prepare_coords(); reused across all
    co-registered images by extract_patches().  Independent of image content.
    """

    row_coords: JArray  # (M, h, w) float32 — source row index per output pixel
    col_coords: JArray  # (M, h, w) float32 — source col index per output pixel


# ---------------------------------------------------------------------------
# Step 1: preprocess mask
# ---------------------------------------------------------------------------


def preprocess_mask(mask: JArray) -> MaskData:
    """
    Preprocess a binary mask into a MaskData object.

    Parameters
    ----------
    mask : (H, W) bool or int — True = valid pixel.

    Returns
    -------
    MaskData — pass to sample_valid_centers().

    Cost: O(H·W).  Call once per mask; reuse across all sampling calls.
    """
    mask = jnp.asarray(mask)
    H, W = int(mask.shape[0]), int(mask.shape[1])
    integral = build_integral_image_jit(mask)
    return MaskData(shape=(H, W), _integral=integral)


# ---------------------------------------------------------------------------
# Step 2: sample valid centers  (no image needed)
# ---------------------------------------------------------------------------


def _batch_aabb_extents(
    patch_hw: Tuple[int, int],
    L_invs: JArray,
) -> tuple[JArray, JArray]:
    h, w = patch_hw
    r_u = (h - 1) / 2.0
    r_v = (w - 1) / 2.0
    L = jnp.asarray(L_invs)
    e_rows = jnp.abs(L[:, 0, 0]) * r_u + jnp.abs(L[:, 0, 1]) * r_v  # (N,)
    e_cols = jnp.abs(L[:, 1, 0]) * r_u + jnp.abs(L[:, 1, 1]) * r_v  # (N,)
    return e_rows, e_cols


_batch_aabb_extents_jit = jx.jit(
    _batch_aabb_extents,
    static_argnames=("patch_hw",),
)


def _check_validity(
    integral: JArray,
    centers: JArray,
    e_rows: JArray,
    e_cols: JArray,
) -> JArray:
    """Vectorised AABB validity check — 4 integral-image lookups per candidate."""
    H = integral.shape[0] - 1
    W = integral.shape[1] - 1

    r = centers[:, 0]
    c = centers[:, 1]

    r0 = jnp.floor(r - e_rows).astype(jnp.int32)
    r1 = jnp.ceil(r + e_rows).astype(jnp.int32)
    c0 = jnp.floor(c - e_cols).astype(jnp.int32)
    c1 = jnp.ceil(c + e_cols).astype(jnp.int32)

    in_bounds = (r0 >= 0) & (r1 <= H - 1) & (c0 >= 0) & (c1 <= W - 1)

    # Clamped indices for safe lookup where !in_bounds.
    r0c = jnp.clip(r0, 0, H)
    r1c = jnp.clip(r1, 0, H - 1)
    c0c = jnp.clip(c0, 0, W)
    c1c = jnp.clip(c1, 0, W - 1)

    I = integral
    window_sum = I[r1c + 1, c1c + 1] - I[r0c, c1c + 1] - I[r1c + 1, c0c] + I[r0c, c0c]
    expected = (r1c - r0c + 1) * (c1c - c0c + 1)

    return in_bounds & (window_sum == expected)


_check_validity_jit = jx.jit(_check_validity)


def sample_valid_centers(
    mask_data: MaskData,
    patch_hw: Tuple[int, int],
    L_invs: JArray,
    key: JArray,
) -> SampledCenters:
    """
    Sample N candidate centers uniformly and return those whose AABB fits the mask.

    Parameters
    ----------
    mask_data : MaskData — from preprocess_mask().
    patch_hw  : (h, w) output patch size.
    L_invs    : (N, 2, 2) float — per-candidate inverse linear maps, sampled by
                the caller from the augmentation distribution.
    key       : JAX PRNGKey (consumed for center sampling).

    Returns
    -------
    SampledCenters with M ≤ N accepted entries.
    Pass .centers and .L_invs to extract_patches() with any co-registered image.

    Cost: O(N) on-device + N-byte host sync.
    """
    H, W = mask_data.shape
    N = L_invs.shape[0]

    # Sample N centers uniformly.
    key_r, key_c = jr.split(key)
    r = jr.uniform(key_r, (N,), minval=0.0, maxval=float(H - 1))
    c = jr.uniform(key_c, (N,), minval=0.0, maxval=float(W - 1))
    centers = jnp.stack([r, c], axis=-1)  # (N, 2)

    # Batched AABB extents and validity check.
    e_rows, e_cols = _batch_aabb_extents_jit(patch_hw, L_invs)
    valid_mask = _check_validity_jit(mask_data._integral, centers, e_rows, e_cols)

    # Filter to accepted candidates (N-byte host sync).
    accepted = np.where(np.array(valid_mask))[0]

    if len(accepted) > 0:
        accepted_centers = centers[accepted]  # (M, 2)
        accepted_L_invs = L_invs[accepted]  # (M, 2, 2)
    else:
        accepted_centers = jnp.zeros((0, 2), dtype=jnp.float32)
        accepted_L_invs = jnp.zeros((0, 2, 2), dtype=jnp.float32)

    return SampledCenters(
        centers=accepted_centers,
        L_invs=accepted_L_invs,
        valid_mask=valid_mask,
    )


# ---------------------------------------------------------------------------
# Step 3: prepare coords  (image-independent)
# ---------------------------------------------------------------------------


def _prepare_coords_batch(
    patch_hw: Tuple[int, int],
    centers: JArray,
    L_invs: JArray,
) -> tuple[JArray, JArray]:
    # vmap center_to_coords over (L_inv, center) pairs.
    result = jax.vmap(functools.partial(center_to_coords, patch_hw))(L_invs, centers)
    return result[0], result[1]  # each (M, h, w)


_prepare_coords_batch_jit = jx.jit(
    _prepare_coords_batch,
    static_argnames=("patch_hw",),
)


def prepare_coords(
    patch_hw: Tuple[int, int],
    centers: JArray,
    L_invs: JArray,
) -> PreparedCoords:
    """
    Precompute source-space sampling coordinates for M patches.

    This is image-independent: call once, then reuse the result across all
    co-registered images via extract_patches().

    Parameters
    ----------
    patch_hw : (h, w) output patch size.
    centers  : (M, 2) float32 — typically sampled.centers from SampledCenters.
    L_invs   : (M, 2, 2) float — typically sampled.L_invs from SampledCenters.

    Returns
    -------
    PreparedCoords — pass to extract_patches() with each source image.

    Cost: O(M·h·w), fully batched in one JIT'd call.
    """
    row_coords, col_coords = _prepare_coords_batch_jit(patch_hw, centers, L_invs)
    return PreparedCoords(row_coords=row_coords, col_coords=col_coords)


# ---------------------------------------------------------------------------
# Step 4: extract patches from an image
# ---------------------------------------------------------------------------


def _extract_one(image: JArray, row_coords: JArray, col_coords: JArray) -> JArray:
    return transform_from_coords(image, [row_coords, col_coords])


# Batched warp over all M patches in a single fused, vectorized call (vmap over
# the patch axis; the image is shared). This replaces a Python loop of M
# per-patch JIT dispatches — the dominant s1_patch_crop cost on CPU.
_extract_patches_batched_jit = jx.jit(jax.vmap(_extract_one, in_axes=(None, 0, 0)))


def extract_patches(
    image: JArray,
    prepared: PreparedCoords,
) -> JArray:
    """
    Apply precomputed coordinates to extract all M patches in one batched call.

    Parameters
    ----------
    image    : (H, W) or (H, W, C) source image.
    prepared : PreparedCoords — from prepare_coords().

    Returns
    -------
    Stacked JArray (M, h, w) or (M, h, w, C). Index ``[i]`` for patch i.

    Cost: O(M·h·w).  Coords are not recomputed — call for each co-registered
    image without repeating prepare_coords().
    """
    return _extract_patches_batched_jit(image, prepared.row_coords, prepared.col_coords)
