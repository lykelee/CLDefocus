from functools import lru_cache

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from datasyn.jaxutils import jx
from datasyn.jaxutils.typing import *


def rgb_to_bgr(x: JArray, c_axis=-1):
    return jnp.flip(x, axis=c_axis)


def bgr_to_rgb(x: JArray, c_axis=-1):
    return jnp.flip(x, axis=c_axis)


def signed_to_rgb(x: JArray, maxval: Optional[float] = None):
    if maxval is None:
        maxval = jnp.max(jnp.abs(x))
    u = x / maxval
    pos = jnp.clip(u, 0.0, 1.0)
    neg = jnp.clip(-u, 0.0, 1.0)
    rgb = jnp.stack([pos, jnp.zeros_like(pos), neg], axis=-1)
    return rgb


def angle_to_hsv(a: JArray):
    """HSV is within [0, 1]"""
    H = jnp.remainder(a / (2 * jnp.pi), 1)
    S = jnp.full_like(H, 1.0)
    V = jnp.full_like(H, 1.0)
    HSV = jnp.stack([H, S, V], axis=-1)
    return HSV


def angle_to_rgb(a: JArray):
    """RGB is within [0, 1]"""
    hsv = angle_to_hsv(a)
    return hsv_to_rgb(hsv)


def complex_to_hsv(cplx: JArray):
    """HSV is within [0, 1]

    NOTE: Currently, inconsistent angle-to-hue with angle_to_hsv!
    """
    phase = jnp.angle(cplx)
    H = (phase + jnp.pi) / (2 * jnp.pi)
    magnitude = jnp.abs(cplx)

    V = magnitude
    # V = jnp.log1p(magnitude)  # log1p(x) = log(1 + x). Use `clip` if needed.

    V_max = jnp.max(V)
    if V_max > 0:
        V = V / V_max
    S = jnp.full_like(H, 1.0)
    HSV = jnp.stack([H, S, V], axis=-1)
    return HSV


def complex_to_rgb(cplx: JArray):
    """RGB is within [0, 1]"""
    hsv = complex_to_hsv(cplx)
    return hsv_to_rgb(hsv)


def complex_to_rgb_np(cplx: NDArray):
    """
    Converts a complex NumPy array into an RGB image array
    using the standard Phase/Magnitude (Domain Coloring) technique.
    """
    import colorsys

    # 1. Calculate Phase (Hue) and Magnitude (Value)

    # Phase: Angles in [-pi, pi].
    # Map to [0, 1] for Hue (H) in HSV space.
    phase = np.angle(cplx)
    H = (phase + np.pi) / (2 * np.pi)

    # Magnitude: Logarithmic scaling is often used to handle large dynamic range.
    # Map to [0, 1] for Value (V) in HSV space.
    magnitude = np.abs(cplx)
    V = np.log1p(magnitude)  # log1p(x) = log(1 + x). Use np.clip if needed.

    # Normalize V to ensure it's in the range [0, 1]
    # We use a simple max-normalization for visualization clarity.
    V_max = np.max(V)
    if V_max > 0:
        V = V / V_max

    # Saturation (S): Set constant high for maximum color visibility.
    # Often set to 1.0 (fully saturated) or a slightly lower value like 0.8
    S = np.full_like(H, 0.9)

    # 2. Prepare for HSV to RGB Conversion

    # Reshape H, S, V into a (H*W, 3) array for efficient conversion
    hsv_flat = np.stack([H.flatten(), S.flatten(), V.flatten()], axis=-1)

    # 3. Perform HSV to RGB Conversion

    # colorsys.hsv_to_rgb is not vectorized, so we apply it element-wise.
    # This is the most computationally intensive step for large arrays.
    # For performance, one might use a custom vectorized function or OpenCV.
    rgb_flat = np.array([colorsys.hsv_to_rgb(h, s, v) for h, s, v in hsv_flat])

    # 4. Reshape and Scale to uint8 (Standard Image Format)

    # Reshape back to the original 2D shape with 3 color channels
    rgb = rgb_flat.reshape(cplx.shape[0], cplx.shape[1], 3)

    return rgb


def rgb_to_hsv(arr: JArray):
    """
    Convert an array of RGB values (assumed in [0, 1]) to HSV, using a JAX-friendly implementation.

    Parameters
    ----------
    arr : (..., 3) array-like
       RGB values in the range [0, 1].

    Returns
    -------
    (..., 3) jnp.ndarray
       HSV values in the range [0, 1].
    """
    arr = jnp.asarray(arr, dtype=jnp.float32)
    if arr.shape[-1] != 3:
        raise ValueError(
            f"Last dimension of input array must be 3; got shape {arr.shape}"
        )

    # Separate channels.
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    # Compute per-pixel maximum and minimum.
    arr_max = jnp.maximum(jnp.maximum(r, g), b)
    arr_min = jnp.minimum(jnp.minimum(r, g), b)
    delta = arr_max - arr_min

    # Compute saturation: avoid division by zero.
    s = jnp.where(arr_max > 0, delta / arr_max, 0.0)

    # Compute hue. If delta is 0, hue is set to 0.
    # Use jnp.select for clarity.
    cond_r = (r == arr_max) & (delta > 0)
    cond_g = (g == arr_max) & (delta > 0)
    cond_b = (b == arr_max) & (delta > 0)
    h_raw = jnp.select(
        [cond_r, cond_g, cond_b],
        [(g - b) / delta, 2.0 + (b - r) / delta, 4.0 + (r - g) / delta],
        default=0.0,
    )
    h = jnp.mod(h_raw / 6.0, 1.0)

    hsv = jnp.stack([h, s, arr_max], axis=-1)
    return hsv


def hsv_to_rgb(hsv: JArray):
    """
    Convert an array of HSV values (assumed in [0, 1]) to RGB using a JAX-friendly implementation.

    Parameters
    ----------
    hsv : (..., 3) array-like
       HSV values in the range [0, 1].

    Returns
    -------
    (..., 3) jnp.ndarray
       RGB values in the range [0, 1].
    """
    hsv = jnp.asarray(hsv, dtype=jnp.float32)
    if hsv.shape[-1] != 3:
        raise ValueError(
            f"Last dimension of input array must be 3; got shape {hsv.shape}"
        )

    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h6 = h * 6.0
    i = jnp.floor(h6).astype(jnp.int32)
    f = h6 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i_mod = i % 6

    # Use jnp.select to choose channels according to the hue sector.
    r = jnp.select(
        [i_mod == 0, i_mod == 1, i_mod == 2, i_mod == 3, i_mod == 4, i_mod == 5],
        [v, q, p, p, t, v],
        default=v,
    )
    g = jnp.select(
        [i_mod == 0, i_mod == 1, i_mod == 2, i_mod == 3, i_mod == 4, i_mod == 5],
        [t, v, v, q, p, p],
        default=v,
    )
    b = jnp.select(
        [i_mod == 0, i_mod == 1, i_mod == 2, i_mod == 3, i_mod == 4, i_mod == 5],
        [p, p, t, v, v, q],
        default=v,
    )

    # For achromatic cases, when s == 0, set r = g = b = v.
    r = jnp.where(s == 0, v, r)
    g = jnp.where(s == 0, v, g)
    b = jnp.where(s == 0, v, b)

    rgb = jnp.stack([r, g, b], axis=-1)
    return rgb


_IMAGENET_MEAN = jnp.array([0.485, 0.456, 0.406])
_IMAGENET_STD = jnp.array([0.229, 0.224, 0.225])


def _reshape_channel_vector(v: JArray, ndim: int, c_axis: int):
    c_axis = c_axis % ndim
    shape = [1] * ndim
    shape[c_axis] = v.shape[0]
    return jnp.reshape(v, shape)


def imagenet_normalize(x: JArray, c_axis=-1):
    """
    Normalize an RGB image/tensor with ImageNet statistics.

    Args:
        x: Array containing RGB values, usually in [0, 1].
        channel_axis: Axis of the 3 RGB channels.

    Returns:
        Normalized array with same shape as x.
    """
    if x.shape[c_axis] != 3:
        raise ValueError(
            f"Expected RGB channel size 3 on axis {c_axis}, got {x.shape[c_axis]}"
        )

    mean = _reshape_channel_vector(_IMAGENET_MEAN, x.ndim, c_axis)
    std = _reshape_channel_vector(_IMAGENET_STD, x.ndim, c_axis)
    return (x - mean) / std


def imagenet_denormalize(x: JArray, c_axis=-1):
    """
    Undo ImageNet normalization for an RGB image/tensor.

    Args:
        x: Normalized RGB array.
        channel_axis: Axis of the 3 RGB channels.

    Returns:
        Denormalized array with same shape as x.
    """
    if x.shape[c_axis] != 3:
        raise ValueError(
            f"Expected RGB channel size 3 on axis {c_axis}, got {x.shape[c_axis]}"
        )

    mean = _reshape_channel_vector(_IMAGENET_MEAN, x.ndim, c_axis)
    std = _reshape_channel_vector(_IMAGENET_STD, x.ndim, c_axis)
    return x * std + mean


#
# Color map
#


@lru_cache(maxsize=None)
def _make_colormap_lut(name: str, n: int) -> jnp.ndarray:
    """
    Build and cache a colormap LUT as a JAX array of shape (n, 3).

    Requires matplotlib only when a new colormap is first requested.
    """
    import matplotlib.cm as cm

    cmap = cm.get_cmap(name, n)
    lut = cmap(np.linspace(0.0, 1.0, n, dtype=np.float32))[:, :3]
    return jnp.asarray(lut, dtype=jnp.float32)


def _move_last_axis_to(rgb: JArray, channel_axis: int) -> jnp.ndarray:
    channel_axis = channel_axis % rgb.ndim
    return jnp.moveaxis(rgb, -1, channel_axis)


@lru_cache(maxsize=None)
def make_colormap(
    name: str,
    n: int = 256,
):
    """
    Return a cached function that maps scalar values in [0, 1] to RGB colors.

    Args:
        name: Matplotlib colormap name, e.g., 'viridis', 'cividis'.
        n: Number of LUT entries.

    Returns:
        A function:
            cmap(x, *, channel_axis=-1, clip=True) -> rgb

        where:
            x: arbitrary-shape scalar array
            rgb: x.shape + (3,) if channel_axis == -1
                 otherwise RGB axis moved to channel_axis
    """
    lut = _make_colormap_lut(name, n)

    @jx.jit
    def cmap(
        x: JArray,
        *,
        c_axis: int = -1,
        clip: bool = True,
    ) -> JArray:
        """
        Apply the cached colormap LUT.

        Args:
            x: Scalar array, typically normalized to [0, 1].
            c_axis: Where to place the RGB channel axis in the output.
                Default: -1, so output shape is (..., 3).
            clip: Whether to clip x into [0, 1].

        Returns:
            RGB array with dtype float32.
        """
        x = jnp.asarray(x)

        if clip:
            x = jnp.clip(x, 0.0, 1.0)

        idx = jnp.round(x * (lut.shape[0] - 1)).astype(jnp.int32)
        rgb = lut[idx]  # shape: x.shape + (3,)

        if c_axis != -1:
            rgb = _move_last_axis_to(rgb, c_axis)

        return rgb

    return cmap


def inverse_depth_colorize(
    depth: JArray, d_min: float = 0.1, d_max: float = float("inf"), cmap="turbo"
):
    inv = 1 / depth
    max_inv = 1 / d_min
    min_inv = 1 / d_max
    inverse_depth_normalized = (inv - min_inv) / (max_inv - min_inv)
    return make_colormap(cmap)(inverse_depth_normalized)
