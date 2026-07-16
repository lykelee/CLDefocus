import jax.numpy as jnp

from datasyn.jaxutils.typing import *


def rot2d(theta):
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return jnp.array([[c, -s], [s, c]])


def scale2d(sx, sy: Optional = None):
    if sy is None:
        sy = sx
    return jnp.array([[sx, 0.0], [0.0, sy]])
