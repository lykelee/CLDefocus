"""
safeop may have multiple meanings:
- NaN-free including intermediates (applied for most cases)
- Safe for gradient calculation

TODO: Develop systematic safeop system!
"""

from functools import partial

import jax.numpy as jnp

from datasyn.jaxutils import jx
from datasyn.optics.typing import *

TPyTree = TypeVar("TPyTree")


class WithDefined(NamedTuple, Generic[TPyTree]):
    value: TPyTree
    defined: TPyTree

    @property
    def v(self):
        return self.value

    @property
    def d(self):
        return self.defined

    @property
    def vd(self):
        return self.v, self.d

    @property
    def dv(self):
        return self.d, self.v

    def __getitem__(self, idx):
        return WithDefined(value=self.value[idx], defined=self.defined[idx])


@jx.jit
def div(a: JArray, b: JArray):
    """
    Calculates a / b.

    NOTE: If a = 0 and b = 0, the result is 0.
    """
    defined = b != 0.0
    y = a / jnp.where(defined, b, 1.0)
    return WithDefined(value=y, defined=defined)


@jx.jit
def sqrt(x: JArray):
    """
    This may raise numerical unstability with autograd near 0.

    TODO: Make grad-safe one!
    """
    defined = x >= 0.0
    x_safe = jnp.where(defined, x, 1.0)
    y = jnp.sqrt(x_safe)
    return WithDefined(value=y, defined=defined)


@partial(jx.jit, static_argnames=("axis", "keepdims"))
def norm(x: JArray, axis: int = -1, keepdims: bool = False):
    y = sqrt(jnp.sum(x**2, axis=axis, keepdims=keepdims))
    return y


@jx.jit
def normdir(x: JArray):
    n, d1 = norm(x)[..., None].vd
    u, d2 = div(x, n).vd
    return WithDefined(value=u, defined=d1 & d2)


class NormAndDir(NamedTuple):
    n: JArray
    u: JArray


def norm_and_dir(x: JArray):
    n, d1 = norm(x)[..., None].vd
    u, d2 = div(x, n).vd
    return WithDefined(value=NormAndDir(n=n[..., 0], u=u), defined=d1 & d2)


@jax.custom_jvp
def gradsafe_norm(x: JArray):
    """
    Satisfies eikonal equation except at zero.
    """
    return norm(x, axis=-1, keepdims=False).v


@gradsafe_norm.defjvp
def _gradsafe_norm_jvp(primals, tangents):
    (p,), (p_dot,) = primals, tangents
    nu, defined = norm_and_dir(p).vd
    rho = nu.n
    u = jnp.where(defined, nu.u, 0)
    # unit normal where rho > 0; zero vector at center
    y_dot = jnp.sum(u * p_dot, axis=-1)
    return rho, y_dot


@jax.custom_jvp
def gradsafe_arctan2(y, x):
    return jnp.arctan2(y, x)


@gradsafe_arctan2.defjvp
def _gradsafe_arctan2_jvp(primals, tangents):
    y, x = primals
    y_dot, x_dot = tangents

    theta = jnp.arctan2(y, x)

    r2 = x * x + y * y
    defined = r2 > 0

    # Avoid division by zero in the tangent rule.
    # At the origin, define the directional derivative to be 0.
    safe_r2 = jnp.where(defined, r2, 1.0)

    theta_dot = (x * y_dot - y * x_dot) / safe_r2
    theta_dot = jnp.where(defined, theta_dot, 0.0)

    return theta, theta_dot


def order_pair(x: JArray, y: JArray):
    """
    TODO: Check if there's a case where this is undefined.
    """
    return jnp.minimum(x, y), jnp.maximum(x, y)


def quadratic_nonzero_a(
    a: float, b: float, c: float, order: bool = False
) -> WithDefined[Tuple[JArray, JArray]]:
    """
    Solves the quadratic equation ax^2 + bx + c = 0 where a is not zero.

    TODO: Ignore `order` if coefficients are complex numbers. (And raise warning)
    """
    disc = b**2 - 4.0 * a * c
    sqrt_disc = sqrt(disc)

    q = -0.5 * (b + jnp.copysign(sqrt_disc.v, b))
    t0 = q / a
    t1 = c / jnp.where(
        q == 0.0, 1.0, q
    )  # q = 0 only if b = c = 0 (so t1 = 0). Avoid calculating 0/0.

    if order:
        t0, t1 = order_pair(t0, t1)

    d = sqrt_disc.d
    return WithDefined((t0, t1), (d, d))


def mul(a: JArray, b: JArray):
    """
    Native 0 * inf produces NaN. This avoids the native multiplication and produces 0.

    NOTE: This assumes that inputs are not NaN (cascade NaN-free).

    TODO: What about integer overflow? => Currently integer issue is not covered, but we might need it in the future as safeop features.
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)

    # Only floating/complex types can have inf; ints don't need masking.
    if jnp.issubdtype(a.dtype, jnp.inexact) and jnp.issubdtype(b.dtype, jnp.inexact):
        kill_a = (b == 0) & jnp.isinf(a)  # places where a must be turned to 0
        kill_b = (a == 0) & jnp.isinf(b)  # places where b must be turned to 0
        a = jnp.where(kill_a, jnp.zeros_like(a), a)
        b = jnp.where(kill_b, jnp.zeros_like(b), b)
        out = a * b
        defined = ~(kill_a | kill_b)

    else:
        out = a * b
        defined = jnp.ones(out.shape, dtype=bool)

    return WithDefined(out, defined)


def dot(a: JArray, b: JArray, axis=-1):
    """
    NaN-free dot / weighted-sum:
      - Terms computed with `safeop.mul`
      - All ±inf terms are *removed* from the finite sum
      - Final rule:
          both +inf and -inf present -> undefined
          only +inf present          -> +inf
          only -inf present          -> -inf
          neither                    -> finite sum
    Works elementwise/broadcasting like jnp.sum(a*b, axis=axis).
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)

    # Fast path for exact types (ints, bools): no inf/NaN possible.
    if not (
        jnp.issubdtype(a.dtype, jnp.inexact) or jnp.issubdtype(b.dtype, jnp.inexact)
    ):
        return jnp.sum(a * b, axis=axis)

    prod, def_prod = mul(a, b).vd

    pos_inf = jnp.isposinf(prod)
    neg_inf = jnp.isneginf(prod)

    # Remove all ±inf terms from the finite accumulation to avoid (+inf)+(-inf) during the sum.
    finite_prod = jnp.where(pos_inf | neg_inf, jnp.zeros((), prod.dtype), prod)
    finite_sum = jnp.sum(finite_prod, axis=axis)

    any_pos = jnp.any(pos_inf, axis=axis)
    any_neg = jnp.any(neg_inf, axis=axis)

    out = jnp.where(
        any_pos & any_neg,
        jnp.zeros_like(finite_sum),
        jnp.where(
            any_pos,
            jnp.array(jnp.inf, finite_sum.dtype),
            jnp.where(any_neg, jnp.array(-jnp.inf, finite_sum.dtype), finite_sum),
        ),
    )
    defined = def_prod & jnp.expand_dims(~(any_pos & any_neg), axis=axis)
    return WithDefined(out, defined)
