import jax
import jax.numpy as jnp

from datasyn.jaxutils.typing import *
from datasyn.mathutils.vecop import normdir, vecnorm


def best_fit_line_intersection(
    p: JArray,  # (N, 3) points on each line
    u: JArray,  # (N, 3) directions (need not be unit; will be normalized)
    w: Optional[JArray] = None,  # (N,) optional weights
    reg: float = 0.0,  # small Tikhonov like 1e-9 if needed
) -> Tuple[JArray, JArray, JArray]:
    """
    Returns:
      x_star : (D,) best-fit intersection point
      residuals_per_line : (N,) perpendicular distances to x_star
      A : (D,D) normal matrix (can be used to gauge conditioning / covariance)
    """
    assert p.ndim == 2
    dim = p.shape[1]  # D
    assert u.shape == p.shape and w.shape == p.shape[:1]

    u_hat = normdir(u)

    if w is None:
        w = jnp.ones(p.shape[0], dtype=p.dtype)
    w = w.reshape(-1, 1, 1)  # broadcast as (N, 1, 1)

    I = jnp.eye(dim, dtype=p.dtype)
    P = (
        I - u_hat[..., None] * u_hat[:, None, :]
    )  # (N, D, D) projectors orthogonal to u_i

    A = jnp.sum(w * P, axis=0)  # (D,D)
    b = jnp.sum((w * P) @ p[:, :, None], axis=0)  # (D, 1)

    if reg > 0:
        A = A + reg * I

    # Fall back to pseudoinverse if singular
    def _solve():
        return jnp.linalg.solve(A, b).squeeze(-1)

    def _pinv():
        return (jnp.linalg.pinv(A) @ b).squeeze(-1)

    x_star = jax.lax.cond(jnp.linalg.cond(A) < 1e8, _solve, _pinv)

    # residual perpendicular distances
    r = (P @ (x_star[None, :, None] - p[:, :, None])).squeeze(-1)  # (N, D)
    residuals = vecnorm(r)  # (N,)

    return x_star, residuals, A
