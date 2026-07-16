import jax.numpy as jnp

from datasyn.jaxutils.typing import *


def vecnorm(x: JArray, ord: int = 2, axis: int = -1, keepdims: bool = False) -> JArray:
    return jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)


def vecnormsqr(x: JArray, axis: int = -1, keepdims: bool = False) -> JArray:
    """This is cheaper than norm because of absence of sqrt operation."""
    return jnp.sum(x**2, axis=axis, keepdims=keepdims)


def vecdot(x: JArray, y: JArray, axis: int = -1):
    return jnp.sum(x * y, axis=axis)


def normdir(x: JArray):
    n = vecnorm(x, axis=-1, keepdims=True)
    n_safe = jnp.where(n == 0.0, 1.0, n)
    u = x / n_safe
    return u


def norm_and_unit(x: JArray):
    n = vecnorm(x, axis=-1, keepdims=True)
    n_safe = jnp.where(n == 0.0, 1.0, n)
    u = x / n_safe
    return n, u


def sqrnorm_and_unit(x: JArray):
    n2 = vecnormsqr(x, axis=-1, keepdims=True)
    n = jnp.sqrt(n2)
    n_safe = jnp.where(n == 0.0, 1.0, n)
    u = x / n_safe
    return n2, u


def dir2d(theta: JArray, axis: int = -1):
    """Unit direction vectors of a given angle array."""
    return jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=axis)


def xy2xyz(xy: JArray, z: JArray | Scalar):
    z = jnp.broadcast_to(z, xy[..., 0:1].shape)
    return jnp.concat([xy, z], axis=-1)
