import jax
import jax.numpy as jnp

from datasyn.jaxutils import jx
from datasyn.jaxutils.typing import *


@jx.jit
def compute_local_scales(S: JArray) -> JArray:
    """Nearest-neighbor based local scales."""
    S = jnp.asarray(S)
    diffs = jnp.diff(S)
    left_gap = jnp.concat([diffs[:1], diffs])
    right_gap = jnp.concat([diffs, diffs[-1:]])
    d = jnp.minimum(left_gap, right_gap)
    return d


def compute_positive_weights(S, a, b, max_interior_mass=0.4):
    """
    NOTE: S should be sorted!

    Compute positive mixture weights over all S such that sum p_i S_i = c,
    where c = (a+b)/2 is assumed to lie strictly inside (S_min, S_max).

    This uses a small positive mass on interior points and adjusts the
    endpoints so the mean equals c. Interior mass is shrunk if needed
    to keep endpoint weights positive.

    NOTE: This is written with Python control flow; you typically call
    it once (not JIT) to precompute weights for a given S,a,b.
    """
    import numpy as np  # use host NumPy for convenience

    # S = np.sort(np.unique(np.asarray(S)))
    N = S.shape[0]
    assert N >= 2, "Need at least two points in S."
    s_min, s_max = S[0], S[-1]
    c = 0.5 * (a + b)
    if not (s_min < c < s_max):
        raise ValueError(
            f"For this positive-weights construction, require S[0] < (a+b)/2 < S[-1]. S[0] = {s_min}, (a+b)/2 = {c}, S[-1] = {s_max}"
        )

    if N == 2:
        denom = s_max - s_min
        p0 = (s_max - c) / denom
        p1 = 1.0 - p0
        return jnp.array([p0, p1])

    N_int = N - 2
    interior_mass = max_interior_mass

    for _ in range(50):
        if N_int > 0:
            q = np.full(N_int, interior_mass / N_int, dtype=float)
            P_int = float(np.sum(q))
            M_int = float(np.sum(q * S[1:-1]))
        else:
            q = np.array([], dtype=float)
            P_int = 0.0
            M_int = 0.0

        denom = s_max - s_min
        p0 = ((1.0 - P_int) * s_max - (c - M_int)) / denom
        pN = 1.0 - P_int - p0

        if p0 > 0 and pN > 0:
            w = np.empty(N, dtype=float)
            w[0] = p0
            w[-1] = pN
            if N_int > 0:
                w[1:-1] = q
            return jnp.asarray(w)

        interior_mass *= 0.5

    raise RuntimeError(
        "Failed to find strictly positive weights; maybe c is too close to an endpoint."
    )


def sample_from_S_with_local_noise(
    key: RngKey,
    S: JArray,
    weights: JArray,
    a: float,
    b: float,
    alpha: float,
    num_samples: int,
) -> JArray:
    """
    Sample from mixture over S with local nearest-neighbor scaled Gaussian noise.

    - key: JAX PRNGKey
    - S: 1D sorted array of shape (N,)
    - weights: shape (N,), positive, sum to 1
    - a, b: bounds for clipping
    - alpha: scale factor for local noise (0.0 -> no noise)
    - num_samples: number of samples
    """
    S = jnp.unique(jnp.asarray(S), sorted=True)
    weights = jnp.asarray(weights)
    N = S.shape[0]

    key_idx, key_noise = jax.random.split(key)
    idx = jax.random.choice(key_idx, N, shape=(num_samples,), p=weights)

    local_scales = compute_local_scales(S)  # shape (N,)
    scales = alpha * local_scales[idx]  # shape (num_samples,)
    eps = jax.random.normal(key_noise, (num_samples,)) * scales
    x = S[idx] + eps
    x = jnp.clip(x, a, b)
    return x
