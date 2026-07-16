from __future__ import annotations

from functools import partial

import jax.numpy as jnp

from datasyn.jaxutils import jx
from datasyn.jaxutils.typing import JArray


@partial(jx.jit, static_argnames=("ks", "normalize"))
def gaussian_kernel(
    ks: int,
    mean: JArray,
    cov: JArray,
    *,
    eps: float = 1e-3,
    normalize: bool = True,
) -> JArray:
    """
    Generate a 2D Gaussian kernel from a mean and 2x2 covariance matrix.

    Coordinate convention: pixel units relative to the array center, with
    x = column and y = row.

    Near-zero covariance is handled by adding ``eps`` to the diagonal before
    inverting. This is smooth everywhere and keeps gradients well-defined
    w.r.t. both ``mean`` and ``cov``. As cov -> 0 the output approaches a
    discrete Kronecker delta centred at ``mean``.

    Args:
        ks: Square kernel side length in pixels.
        mean: (..., 2) — [mu_x, mu_y] in pixel coords from array center.
        cov: (..., 2, 2) — spatial covariance matrix (must be positive
            semi-definite; eps regularisation handles the degenerate case).
        eps: Diagonal regularisation added to cov before inversion.
            Keeps the effective minimum variance at eps pixels².

    Returns:
        kernel: (..., ks, ks).
    """
    coords = jnp.arange(ks, dtype=jnp.float32) - (ks - 1) * 0.5  # (ks,)

    mu_x = mean[..., 0, None, None]  # (..., 1, 1)
    mu_y = mean[..., 1, None, None]  # (..., 1, 1)
    dx = coords[None, :] - mu_x  # (..., 1, ks)
    dy = coords[:, None] - mu_y  # (..., ks, 1)

    a = cov[..., 0, 0] + eps  # (...,)
    b = cov[..., 0, 1]  # (...,)
    d = cov[..., 1, 1] + eps  # (...,)

    det = a * d - b * b
    inv_a = d / det
    inv_b = -b / det
    inv_d = a / det

    maha2 = (
        inv_a[..., None, None] * dx * dx
        + 2.0 * inv_b[..., None, None] * dx * dy
        + inv_d[..., None, None] * dy * dy
    )

    kernel = jnp.exp(-0.5 * maha2)

    if normalize:
        kernel = kernel / kernel.sum(axis=(-2, -1), keepdims=True)

    return kernel
