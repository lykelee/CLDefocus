from __future__ import annotations

from typing import Literal, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.ndimage import map_coordinates

from datasyn.jaxutils import jx
from datasyn.jaxutils.typing import *

Array = jnp.ndarray


def _centered_grid(h: int, w: int, dtype=jnp.float32) -> tuple[Array, Array]:
    """Return (yy, xx) grids of shape (h, w) centered at 0 in pixel-index space.

    Pixel centers are at integer coordinates. For even sizes, the center is half-integer.
    """
    yy = jnp.arange(h, dtype=dtype) - (h - 1) / 2.0
    xx = jnp.arange(w, dtype=dtype) - (w - 1) / 2.0
    yy, xx = jnp.meshgrid(yy, xx, indexing="ij")  # yy: row offsets, xx: col offsets
    return yy, xx


def _safe_uniform(key: Array, lo: Array, hi: Array) -> Array:
    """Sample U[lo, hi) if hi >= lo; otherwise return midpoint (lo+hi)/2.

    This avoids NaNs when the valid interval is empty. (Assumes source is "usually large enough".)
    """
    mid = 0.5 * (lo + hi)
    u = jr.uniform(key, shape=(), minval=0.0, maxval=1.0)
    samp = lo + (hi - lo) * u
    return jnp.where(hi >= lo, samp, mid)


def affine_aabb_extents(
    patch_hw: Tuple[int, int],
    linear_inv: Array,
) -> tuple[Array, Array]:
    """
    Half-extents of the axis-aligned bounding box of the source-space parallelogram.

    For a patch of size (h, w) warped by inverse linear map L, the parallelogram
    footprint in source coordinates spans [c_row ± e_row] × [c_col ± e_col].

    Returns
    -------
    e_row, e_col : JAX scalars — AABB half-extents in (row, col) directions.
    """
    h, w = patch_hw
    r_u = (h - 1) / 2.0
    r_v = (w - 1) / 2.0
    L = jnp.asarray(linear_inv)
    e_row = jnp.abs(L[0, 0]) * r_u + jnp.abs(L[0, 1]) * r_v
    e_col = jnp.abs(L[1, 0]) * r_u + jnp.abs(L[1, 1]) * r_v
    return e_row, e_col


affine_aabb_extents_jit = jx.jit(
    affine_aabb_extents,
    static_argnames=("patch_hw",),
)


def sample_affine_patch(
    image_hw: Tuple[int, int],
    patch_hw: Tuple[int, int],
    linear_inv: Array,
    key: Array,
    *,
    eps: float = 1e-3,
):
    """
    Sample a border-safe affine-augmented patch using inverse warping + map_coordinates.

    Coordinate convention (pixel-index space):
      - image is indexed as image[row, col, ...]
      - pixel centers are at integer coordinates
      - source valid domain is row in [0, H-1], col in [0, W-1]

    We generate the final patch (h, w) by:
      x_src(y) = c + L @ y
    where y = [row_offset, col_offset]^T is centered output-grid coordinate,
    c is the sampled source coordinate for the output patch center,
    and L is the linear map from output offsets to source offsets.

    Args
    ----
    image:
      (H, W) or (H, W, C). (You can vmap/batch outside this function.)
    linear:
      2x2 matrix. If linear_kind="out_from_in", this is M in y = M x (source->output).
      If linear_kind="in_from_out", this is L in x = L y (output->source).
    patch_hw:
      (h, w) of the output patch.
    key:
      PRNGKey.
    linear_kind:
      "out_from_in" or "in_from_out".
    order:
      Interpolation order passed to map_coordinates (JAX supports 0 or 1).
    mode, cval:
      Boundary behavior for map_coordinates (should be irrelevant if sampling is truly in-bounds).
    eps:
      Shrink the valid center range a bit to avoid hitting boundaries due to float roundoff.
    return_center:
      If True, also return the sampled center c as shape (2,) = (row_center, col_center).

    Returns
    -------
    patch:
      (h, w) or (h, w, C)
    (optional) center:
      (2,) float array (row_center, col_center)
    """
    H, W = image_hw
    e_row, e_col = affine_aabb_extents(patch_hw, linear_inv)

    # Valid ranges for the source-center c = (c_row, c_col)
    lo_row = e_row + eps
    hi_row = (H - 1) - e_row - eps
    lo_col = e_col + eps
    hi_col = (W - 1) - e_col - eps

    key_row, key_col = jr.split(key, 2)
    c_row = _safe_uniform(key_row, lo_row, hi_row)
    c_col = _safe_uniform(key_col, lo_col, hi_col)
    c = jnp.stack([c_row, c_col], axis=0)  # (2,)

    return c


sample_affine_patch_jit = jx.jit(
    sample_affine_patch,
    static_argnames=("image_hw", "patch_hw"),
)


def center_to_coords(
    patch_hw: Tuple[int, int], M_inv: JArray, patch_cen: Tuple[int, int]
):
    h, w = patch_hw
    M_inv = jnp.asarray(M_inv)

    # Output patch centered grid (row/col offsets)
    yy, xx = _centered_grid(h, w)

    # Source coordinates for each output pixel:
    src_row = patch_cen[0] + M_inv[0, 0] * yy + M_inv[0, 1] * xx
    src_col = patch_cen[1] + M_inv[1, 0] * yy + M_inv[1, 1] * xx
    coords = [src_row, src_col]

    return coords


center_to_coords_jit = jx.jit(
    center_to_coords,
    static_argnames=("patch_hw",),
)


def transform_from_coords(
    image: Array,
    coords: list[Array],
    *,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
) -> Array:
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be (H,W) or (H,W,C), got shape {image.shape}")

    if image.ndim == 2:
        patch = map_coordinates(image, coords, order=order, mode=mode, cval=cval)

    else:
        # Warp each channel independently (no cross-channel interpolation).
        img_chw = jnp.moveaxis(image, -1, 0)  # (C, H, W)

        def warp_one(ch_img: Array) -> Array:
            return map_coordinates(ch_img, coords, order=order, mode=mode, cval=cval)

        patch_chw = jx.vmap(warp_one, in_axes=0, out_axes=0)(img_chw)  # (C, h, w)
        patch = jnp.moveaxis(patch_chw, 0, -1)  # (h, w, C)

    return patch


transform_from_coords_jit = jx.jit(
    transform_from_coords,
    static_argnames=("order", "mode"),
)


BlueMode = Literal["diag_sine", "radial", "diag"]


def uv_grid_calibration(
    height: int,
    width: int,
    *,
    cell_size: Union[int, float] = 32,
    major_step: int = 4,
    minor_line_px: Union[int, float] = 1.0,
    major_line_px: Union[int, float] = 3.0,
    corner_frac: float = 0.32,
    corner_color: Tuple[float, float, float] = (1.0, 0.0, 1.0),  # magenta
    corner_alpha: float = 0.95,
    checker_strength: float = 0.22,
    blue_mode: BlueMode = "diag_sine",
    base_offset: float = 0.08,
    base_scale: float = 0.92,
    out_dtype=jnp.uint8,  # set to jnp.float32 for [0,1] output
) -> jnp.ndarray:
    """
    TODO: This sucks! I want to redesign a calibration image in the future.
          The philosophy is as follows:

          - Goal: We apply deformations to the image and take small viewports.
          - Grid-like image
          - Each cell is small enough. In a patch, we expect we find at least several cells that are fully contained in a viewport.
          - We should easily notice location changes.
          - We should easily notice deformation (e.g., rotation, scaling).

    Generate a purely-analytic UV-like calibration image (H,W,3).

    Design goals (per your revisions):
      - Each cell has uniform color (global UV per-cell)
      - Rough position tendency is visible (global color field + major grid cadence)
      - No text, no raster fonts, no per-cell gradients, no arrows
      - Corner marker triangle in a consistent corner to expose orientation/handedness
      - Minor + major grid lines

    Notes:
      - For jitting, `height` and `width` should be static (known at trace time).
      - `cell_size` is in pixels; non-divisible sizes are fine (it just creates partial cells at edges).
    """
    cell_size = jnp.asarray(cell_size, dtype=jnp.float32)
    H = int(height)
    W = int(width)

    # Pixel centers in "cell coordinates"
    yy = jnp.arange(H, dtype=jnp.float32) + 0.5
    xx = jnp.arange(W, dtype=jnp.float32) + 0.5
    y, x = jnp.meshgrid(yy, xx, indexing="ij")  # (H,W)

    tx = x / cell_size
    ty = y / cell_size

    ix = jnp.floor(tx).astype(jnp.int32)
    iy = jnp.floor(ty).astype(jnp.int32)
    ix_f = ix.astype(jnp.float32)
    iy_f = iy.astype(jnp.float32)

    u = tx - ix_f  # in [0,1)
    v = ty - iy_f  # in [0,1)

    # "Global UV" per cell (piecewise constant within each cell)
    tiles_x = jnp.asarray(W, jnp.float32) / cell_size
    tiles_y = jnp.asarray(H, jnp.float32) / cell_size
    cx = (ix_f + 0.5) / tiles_x
    cy = (iy_f + 0.5) / tiles_y

    # Blue channel options (use lax.switch for JIT-friendliness)
    def b_diag_sine(args):
        cx, cy = args
        return 0.5 + 0.5 * jnp.sin(2.0 * jnp.pi * (0.75 * cx + 0.55 * cy))

    def b_radial(args):
        cx, cy = args
        r = jnp.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2) / jnp.sqrt(0.5**2 + 0.5**2)
        return jnp.clip(r, 0.0, 1.0)

    def b_diag(args):
        cx, cy = args
        return 0.5 * (cx + cy)

    mode_to_idx = {"diag_sine": 0, "radial": 1, "diag": 2}
    mode_idx = mode_to_idx.get(blue_mode, 0)
    b = jax.lax.switch(mode_idx, (b_diag_sine, b_radial, b_diag), (cx, cy))

    uv = jnp.stack([cx, cy, b], axis=-1)  # (H,W,3)

    # Cell-level checker luminance modulation (still uniform inside a cell)
    checker = ((ix + iy) & 1).astype(jnp.float32)
    lum = 1.0 + checker_strength * (2.0 * checker - 1.0)  # [1-s, 1+s]
    rgb = jnp.clip(base_offset + base_scale * uv * lum[..., None], 0.0, 1.0)

    # Contrast-aware line color: white on dark areas, black on light areas
    intensity = rgb.mean(axis=-1)
    line_bw = (intensity < 0.5).astype(jnp.float32)  # 1 => white, 0 => black
    line_col = jnp.stack([line_bw, line_bw, line_bw], axis=-1)

    # Edge thickness specified in pixels -> convert to cell-relative thickness
    t_minor = jnp.asarray(minor_line_px, jnp.float32) / cell_size
    t_major = jnp.asarray(major_line_px, jnp.float32) / cell_size

    # Minor edges: all cell boundaries
    minor_edge = (
        jnp.minimum(jnp.minimum(u, 1.0 - u), jnp.minimum(v, 1.0 - v)) < t_minor
    ).astype(jnp.float32)

    # Major edges: every `major_step` cells (thicker)
    ms = int(major_step)

    left_major = ((u < t_major) & ((ix % ms) == 0)).astype(jnp.float32)
    right_major = (((1.0 - u) < t_major) & (((ix + 1) % ms) == 0)).astype(jnp.float32)
    top_major = ((v < t_major) & ((iy % ms) == 0)).astype(jnp.float32)
    bot_major = (((1.0 - v) < t_major) & (((iy + 1) % ms) == 0)).astype(jnp.float32)
    major_edge = jnp.clip(left_major + right_major + top_major + bot_major, 0.0, 1.0)

    # Draw minor then major (major slightly blue-tinted to stand out)
    rgb = rgb * (1.0 - 0.85 * minor_edge[..., None]) + line_col * (
        0.85 * minor_edge[..., None]
    )
    major_tint = jnp.stack([line_bw, line_bw, jnp.ones_like(line_bw)], axis=-1)
    rgb = rgb * (1.0 - 0.95 * major_edge[..., None]) + major_tint * (
        0.95 * major_edge[..., None]
    )

    # Corner marker triangle (same corner in each cell)
    k = jnp.asarray(corner_frac, jnp.float32)
    tri = ((u < k) & (v < k) & ((u + v) < k)).astype(jnp.float32)
    cc = jnp.asarray(corner_color, jnp.float32)
    rgb = rgb * (1.0 - corner_alpha * tri[..., None]) + cc * (
        corner_alpha * tri[..., None]
    )

    if out_dtype == jnp.uint8:
        return (jnp.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(jnp.uint8)
    else:
        return rgb.astype(out_dtype)
