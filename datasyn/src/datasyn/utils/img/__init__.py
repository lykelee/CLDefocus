import jax.numpy as jnp

import datasyn.optics.safeop as safeop
from datasyn.jaxutils.typing import *


def add_patch_drop_outside(
    img: JArray, patch: JArray, top_left: tuple[int, int]
) -> JArray:
    y0, x0 = top_left
    h, w = patch.shape[:2]
    ys = jnp.arange(h) + y0
    xs = jnp.arange(w) + x0
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    if img.ndim == 2:
        return img.at[yy, xx].add(patch, mode="drop")
    elif img.ndim == 3:
        return img.at[yy, xx, :].add(patch, mode="drop")
    else:
        raise ValueError("Only 2D or 3D (H,W[,C]) images are supported.")


def remove_black_to_transparent(rgb: jnp.ndarray) -> jnp.ndarray:
    alpha = jnp.full(rgb.shape[:-1], 1.0, dtype=rgb.dtype)
    black_mask = jnp.all(rgb == 0, axis=-1)
    alpha = alpha.at[black_mask].set(0)
    return jnp.dstack((rgb, alpha))
