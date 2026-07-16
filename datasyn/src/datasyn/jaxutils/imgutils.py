import itertools

import jax.numpy as jnp
import numpy as np
import PIL
import PIL.Image

from datasyn.jaxutils.utils import *


def arr1f_to_pil(x):
    return PIL.Image.fromarray(
        np.clip(255.0 * np.asarray(x), 0.0, 255.0).astype(np.uint8)
    )


def make_grid(
    patches: jnp.ndarray,
    grid_ndim: Optional[int] = None,
    border_width: Optional[int] = 0,
    border_color: Optional[Any] = 0,
):
    """
    Generates an n-dimensional grid image from an n-dimensional grid of n-dimensional patches,
    with extra axes preserved. That is, given an array of shape:

        (a_1, ..., a_n, b_1, ..., b_n, c_1, ..., c_m)

    where:
      • (a_1, ..., a_n) are grid dimensions,
      • (b_1, ..., b_n) are patch dimensions, and
      • (c_1, ..., c_m) are extra axes (for example, color channels),

    the output image has shape
        (a_1 * b_1 + (a_1 + 1) * border_width, ..., a_n * b_n + (a_n + 1) * border_width, c_1, ..., c_m)
    with the patches placed in the grid separated by a border of the specified width and color.

    Args:
        patches: jnp.ndarray of shape (a_1, ..., a_n, b_1, ..., b_n, c_1, ..., c_m)
        border_width: int, the width of the border. If 0, patches are simply rearranged.
        border_color: a scalar or an array with shape (c_1, ..., c_m) specifying the border color.
                      (It must be broadcastable to the extra axes of the patches.)
        grid_ndim: int, optional number of grid (and patch) dimensions (n). If not provided,
                   an attempt is made to infer it using the rank of border_color.

    Returns:
        A jnp.ndarray representing the grid image with extra axes preserved.
    """
    patches = jnp.asarray(patches)

    # Infer grid_ndim if not provided.
    if grid_ndim is None:
        try:
            bc_ndim = border_color.ndim  # if border_color is array-like
        except AttributeError:
            bc_ndim = 0
        remaining = patches.ndim - bc_ndim
        if remaining % 2 == 0:
            grid_ndim = remaining // 2
        else:
            # Fallback: assume no extra axes if inference is ambiguous.
            grid_ndim = patches.ndim // 2

    n = grid_ndim
    grid_shape = patches.shape[:n]  # (a_1, ..., a_n)
    patch_shape = patches.shape[n : 2 * n]  # (b_1, ..., b_n)
    extra_shape = patches.shape[2 * n :]  # (c_1, ..., c_m)  (m can be 0)

    if border_width <= 0:
        # When no border is added, we simply interleave grid and patch axes.
        # The permutation interleaves each grid axis with its corresponding patch axis.
        perm = []
        for i in range(n):
            perm.append(i)
            perm.append(i + n)
        # Append the extra axes (if any) as they are.
        perm.extend(range(2 * n, patches.ndim))
        new_shape = (
            tuple(grid_shape[i] * patch_shape[i] for i in range(n)) + extra_shape
        )
        return patches.transpose(*perm).reshape(new_shape)

    # Compute new spatial shape for each dimension d: a_d * b_d + (a_d+1) * border_width.
    new_spatial_shape = tuple(
        grid_shape[d] * patch_shape[d] + (grid_shape[d] + 1) * border_width
        for d in range(n)
    )
    new_shape = new_spatial_shape + extra_shape

    # Create the output image filled with border_color.
    # Using multiplication by ones helps when border_color is non-scalar.
    image = jnp.ones(new_shape, dtype=patches.dtype) * border_color

    # Iterate over all grid indices.
    for grid_idx in itertools.product(*(range(g) for g in grid_shape)):
        # Compute the starting index in each spatial dimension.
        start_indices = [
            border_width + grid_idx[d] * (patch_shape[d] + border_width)
            for d in range(n)
        ]
        # Build a tuple of slices for spatial dimensions.
        spatial_slices = tuple(
            slice(start, start + patch_shape[d])
            for d, start in enumerate(start_indices)
        )
        # Append slices for the extra axes.
        full_slices = spatial_slices + (slice(None),) * len(extra_shape)
        # Set the corresponding patch into the output image.
        image = image.at[full_slices].set(patches[grid_idx])

    return image


def best_grid_2d(N: int, R: float, band: float = 4.0) -> Tuple[int, int]:
    """
    TODO: Find a better algorithm!

    Finds row and column size to plot N data as a grid, aiming to aspect R.
    This allows padding, because some N doesn't allow good exact grid (e.g., large prime number like 97).
    """
    import math

    if N <= 0 or R <= 0:
        raise ValueError("N and R must be positive")

    a_star = math.sqrt(N * R)
    a_min = max(1, int(a_star / band))
    a_max = max(a_min, int(a_star * band))

    best_score = float("inf")
    best_pair = (1, N)

    for a in range(a_min, a_max + 1):
        b = (N + a - 1) // a  # ceil(N / a)
        area = a * b

        padding_factor = area / N
        ratio = a / b
        shape_factor = max(ratio, R) / min(ratio, R)
        score = padding_factor * shape_factor

        if score < best_score:
            best_score = score
            best_pair = (a, b)

    return best_pair


def make_image_batch_2d_grid(
    image_batch: JArray,
    aspect: float = 1.0,
    border_width: int = 0,
    border_color=None,
    pad_value: float = 0.0,
):
    from datasyn.jaxutils.nputils import chunk

    N = image_batch.shape[0]
    h, w = best_grid_2d(N, R=aspect)
    image_batch_grid = chunk(image_batch, chunk_size=w, pad_value=pad_value)
    gridimg = make_grid(
        image_batch_grid, border_width=border_width, border_color=border_color
    )
    return gridimg
