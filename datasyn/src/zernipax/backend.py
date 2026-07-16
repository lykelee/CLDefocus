"""Backend functions for zernipax."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import config as jax_config
from jax import custom_jvp, jit, vmap  # noqa: F401
from jax.lax import cond, fori_loop, scan, select, switch, while_loop  # noqa: F401
from jax.numpy import bincount  # noqa: F401
from jax.scipy.special import gammaln  # noqa: F401

jax_config.update("jax_enable_x64", True)


def put(arr, inds, vals):
    """Functional interface for array "fancy indexing".

    Provides a way to do arr[inds] = vals in a way that works with JAX.

    Parameters
    ----------
    arr : array-like
        Array to populate
    inds : array-like of int
        Indices to populate
    vals : array-like
        Values to insert

    Returns
    -------
    arr : array-like
        Input array with vals inserted at inds.

    """
    if isinstance(arr, np.ndarray):
        arr[inds] = vals
        return arr
    return jnp.asarray(arr).at[inds].set(vals)


def sign(x):
    """Sign function, but returns 1 for x==0.

    Parameters
    ----------
    x : array-like
        array of input values

    Returns
    -------
    y : array-like
        1 where x>=0, -1 where x<0

    """
    x = jnp.atleast_1d(x)
    y = jnp.where(x == 0, 1, jnp.sign(x))
    return y


def custom_jvp_with_jit(func):
    """Decorator for custom_jvp with jit.

    This decorator is specifically with functions that have the same
    structure as the zernike_radial such as r, l, m, dr, where dr is
    the static argument.
    """

    @functools.partial(
        custom_jvp,
        nondiff_argnums=(3,),
    )
    def dummy(r, l, m, dr=0):
        return func(r, l, m, dr)

    @dummy.defjvp
    def _dummy_jvp(nondiff_dr, x, xdot):
        """Custom derivative rule for the function.

        This is just the same function called with dx+1.
        """
        (r, l, m) = x
        (rdot, ldot, mdot) = xdot
        f = dummy(r, l, m, nondiff_dr)
        df = dummy(r, l, m, nondiff_dr + 1)
        return f, (df.T * rdot).T

    return jit(dummy, static_argnums=3)


def execute_on_cpu(func):
    """Decorator to set default device to CPU for a function.

    Parameters
    ----------
    func : callable
        Function to decorate

    Returns
    -------
    wrapper : callable
        Decorated function that will run always on CPU even if
        there are available GPUs.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with jax.default_device(jax.devices("cpu")[0]):
            return func(*args, **kwargs)

    return wrapper
