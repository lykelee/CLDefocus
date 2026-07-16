import sys

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    ParamSpec,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeAlias,
    TypeVar,
    Union,
)

if sys.version_info < (3, 8):
    pass
else:
    pass

if sys.version_info < (3, 10):
    pass
else:
    pass

# JAX
import jax

from jax import Array
from jax import Device as JaxDevice
from jax.typing import ArrayLike as JArrayLike
from jax.typing import DTypeLike

JArray: TypeAlias = Array

Dtype = Union[DTypeLike, Any]
PyTree: TypeAlias = Any
RngKey: TypeAlias = JArrayLike


# Using jaxtyping

import jaxtyping as jtp

# By types
IntArray: TypeAlias = jtp.Int[jtp.Array, "..."]
BoolArray: TypeAlias = jtp.Bool[jtp.Array, "..."]
FloatArray: TypeAlias = jtp.Float[jtp.Array, "..."]
FloatScalar: TypeAlias = jtp.Float[jtp.Array, ""]
ComplexArray: TypeAlias = jtp.Complex[jtp.Array, "..."]
ComplexScalar: TypeAlias = jtp.Complex[jtp.Array, ""]

Scalar: TypeAlias = jtp.Shaped[jtp.Array, ""]
