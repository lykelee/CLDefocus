"""
TODO: This module is quite messy!
      Reorganize this, separating random and deterministic samplings.
"""

from datasyn.jaxutils.utils import *


def random_circle_interior(rng: RngKey, n: int):
    rng_r, rng_theta = jax.random.split(rng)

    u = jax.random.uniform(rng_r, (n,))
    r = jnp.sqrt(u)

    theta = jax.random.uniform(rng_theta, (n,), minval=0.0, maxval=2 * jnp.pi)

    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    return jnp.stack([x, y], axis=-1)


def map_square_to_disk(uv: JArray, eps: float = 1e-12):
    """
    Maps a square region [-1, 1]^2 to a circle ||x|| = 1 with uniform density.
    This is particularly useful to convert stratified sampling on a square to a circle.
    Refer to https://www.josswhittle.com/concentric-disk-sampling/

    uv: array (..., 2) with values in [-1, 1].
    returns: array (..., 2) uniformly distributed in unit disk.
    """
    uv = jnp.asarray(uv)
    dtype = uv.dtype
    pi = jnp.asarray(jnp.pi, dtype)
    pi_4 = pi / jnp.asarray(4.0, dtype)
    pi_2 = pi / jnp.asarray(2.0, dtype)
    eps = jnp.asarray(eps, dtype)

    sx, sy = uv[..., 0], uv[..., 1]

    # Mask for the origin (measure-zero, but avoid 0/0 and clean gradients there)
    zero = (jnp.abs(sx) + jnp.abs(sy)) < eps

    ax, ay = jnp.abs(sx), jnp.abs(sy)
    use_x = ax > ay  # dominant axis

    # Safe denominators: keep ratios bounded and avoid NaNs on axes
    denom_x = jnp.where(ax < eps, jnp.ones_like(sx), sx)
    denom_y = jnp.where(ay < eps, jnp.ones_like(sy), sy)

    # r takes the SIGNED dominant coordinate to land in the right quadrant
    r = jnp.where(use_x, sx, sy)

    # theta is a linear function of the slope in each sector
    theta_x = pi_4 * (sy / denom_x)  # when |sx| > |sy|
    theta_y = pi_2 - pi_4 * (sx / denom_y)  # otherwise
    theta = jnp.where(use_x, theta_x, theta_y)

    # Polar -> Cartesian
    c, s_ = jnp.cos(theta), jnp.sin(theta)
    x = r * c
    y = r * s_

    out = jnp.stack([x, y], axis=-1)
    # Snap exact origin to (0,0) (useful for reproducibility & clean grads)
    out = jnp.where(zero[..., None], jnp.zeros_like(out), out)
    return out


def _parse_bounds(
    bounds: Optional[Union[Tuple[JArray, JArray], JArray]], d: int, dtype
) -> Tuple[JArray, JArray]:
    """Normalize bounds to (low, high) each with shape (d,)."""
    if bounds is None:
        low = jnp.zeros((d,), dtype=dtype)
        high = jnp.ones((d,), dtype=dtype)
        return low, high
    if isinstance(bounds, tuple) and len(bounds) == 2:
        low, high = bounds
        return jnp.asarray(low, dtype=dtype).reshape((d,)), jnp.asarray(
            high, dtype=dtype
        ).reshape((d,))
    b = jnp.asarray(bounds, dtype=dtype)
    assert b.shape == (d, 2)
    return b[:, 0], b[:, 1]


def stratified_samples(
    key: JArray,
    grid_shape: Iterable[int],
    bounds: Optional[Union[Tuple[JArray, JArray], JArray]] = None,
    *,
    per_cell: int = 1,
    jitter: bool = True,
    keep_structure: bool = False,
    shuffle: bool = False,
    dtype=jnp.float32,
) -> JArray:
    """
    Stratified sampling on a d-D hypercube partitioned into a regular grid.

    Returns
    -------
    samples:
        - if keep_structure=False: shape (prod(grid_shape) * per_cell, d)
        - if keep_structure=True:  shape (*grid_shape, per_cell, d)

    Notes
    -----
    - When keep_structure=True, `shuffle` is not allowed (raises ValueError).
    - Gradients w.r.t. `bounds` are fine; RNG draws are treated as constants.
    """
    grid_shape = tuple(int(n) for n in grid_shape)
    d = len(grid_shape)
    if per_cell <= 0:
        raise ValueError("per_cell must be >= 1")
    if keep_structure and shuffle:
        raise ValueError("shuffle is incompatible with keep_structure=True.")

    low, high = _parse_bounds(bounds, d, dtype)
    cell_size = (high - low) / jnp.asarray(grid_shape, dtype=dtype)

    # Per-axis integer indices as an n-D grid, then stack -> shape (*grid_shape, d)
    ax = [jnp.arange(n, dtype=dtype) for n in grid_shape]
    mesh = jnp.meshgrid(*ax, indexing="ij")
    idx_nd = jnp.stack(mesh, axis=-1)  # (*grid_shape, d)

    # Base corner of each cell: low + idx * cell_size
    base = low + idx_nd * cell_size  # (*grid_shape, d)

    # Offsets inside each cell: shape (*grid_shape, per_cell, d)
    if jitter:
        key, k = jax.random.split(key)
        off = jax.random.uniform(k, shape=grid_shape + (per_cell, d), dtype=dtype)
    else:
        off = jnp.full(grid_shape + (per_cell, d), 0.5, dtype=dtype)

    samples = base[..., None, :] + off * cell_size  # broadcast cell_size (d,)

    if keep_structure:
        return samples  # (*grid_shape, per_cell, d)

    # Flatten to (N * per_cell, d)
    samples = samples.reshape((-1, d))

    if shuffle:
        key, kperm = jax.random.split(key)
        perm = jax.random.permutation(kperm, samples.shape[0])
        samples = samples[perm]

    return samples
