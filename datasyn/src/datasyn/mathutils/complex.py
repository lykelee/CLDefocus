import jax.numpy as jnp

from datasyn.jaxutils import jx
from datasyn.jaxutils.typing import *


@jx.jit
def sqabs_complex(x: ComplexArray) -> FloatArray:
    """Squared absolute value of a complex-valued array."""
    # As far as I know, this is the most efficient way.
    # Using `abs` or `conj` might have slight overhead.
    x = jnp.asarray(x)
    return x.real**2 + x.imag**2


@jx.jit
def abs_complex(x: ComplexArray) -> FloatArray:
    """Absolute value of a complex-valued array."""
    return jnp.sqrt(sqabs_complex(x))


@jx.jit
def dtype_to_complex(dtype: jnp.dtype) -> jnp.dtype:
    """Converts a dtype to a complex dtype, keeping the precision as much as possible."""
    return jnp.promote_types(dtype, jnp.complex64)


@jx.jit
def float2complex(x: FloatArray) -> ComplexArray:
    """Converts a real-valued array to a complex-valued array, keeping the precision as much as possible."""
    x = jnp.asarray(x)
    if dtype is None:
        dtype = jnp.promote_types(x.dtype, jnp.complex64)
    return x.astype(dtype)
