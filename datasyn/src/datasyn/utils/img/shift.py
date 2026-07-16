import jax.numpy as jnp
import jax.scipy.ndimage as jnd

import datasyn.optics.safeop as safeop
from datasyn.jaxutils.typing import *


def compute_centroid(img: JArray):
    """
    img: (H, W) float array
    returns (cy, cx)
    """
    H, W = img.shape

    y = jnp.arange(H)
    x = jnp.arange(W)
    yy, xx = jnp.meshgrid(y, x, indexing="ij")

    mass = jnp.sum(img)

    cy = safeop.div(jnp.sum(yy * img), mass).v
    cx = safeop.div(jnp.sum(xx * img), mass).v

    return cy, cx


def shift_image_subpixel(
    img: JArray, dy, dx, *, order=1, mode="constant", cval=0.0
) -> JArray:
    """
    Subpixel shift using interpolation.
    img: (H, W)
    dy, dx: float shifts (positive dy moves content down in y; positive dx moves content right in x)
    """
    H, W = img.shape
    y = jnp.arange(H, dtype=jnp.float32)
    x = jnp.arange(W, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(y, x, indexing="ij")

    # Sample input at inverse-warp coordinates
    coords = jnp.stack([yy - dy, xx - dx], axis=0)  # (2, H, W)

    # order=1 -> bilinear, mode='constant' fills outside with cval
    return jnd.map_coordinates(img, coords, order=order, mode=mode, cval=cval)


def align_to_centroid_subpixel(img: JArray, *, order=1, mode="constant", cval=0.0):
    """
    Align intensity centroid to geometric center with subpixel interpolation.
    """
    H, W = img.shape
    cy, cx = compute_centroid(img)

    target_y = (H - 1) / 2.0
    target_x = (W - 1) / 2.0

    dy = target_y - cy
    dx = target_x - cx

    return shift_image_subpixel(img, dy, dx, order=order, mode=mode, cval=cval)


def align_to_centroid_dft_layout(img: JArray, *, order=1, mode="constant", cval=0.0):
    """
    NOTE: This is quick-and-dirty for my defocus blur synthesis blur kernels.
    """
    H, W = img.shape
    cy, cx = compute_centroid(img)

    target_y = H // 2
    target_x = W // 2

    dy = target_y - cy
    dx = target_x - cx

    return shift_image_subpixel(img, dy, dx, order=order, mode=mode, cval=cval)
