"""
Minimal typed shims over native jax / jax.lax.
"""

from __future__ import annotations

from typing import overload

try:
    from typing import TypeVarTuple, Unpack
except ImportError:  # Python < 3.11
    from typing_extensions import TypeVarTuple, Unpack

import jax
import jax.lax as lax

from datasyn.jaxutils.typing import (
    Any,
    Callable,
    Iterable,
    JArray,
    JaxDevice,
    ParamSpec,
    Sequence,
    Tuple,
    TypeVar,
)

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")
TOut = TypeVar("TOut")
Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")
Ts = TypeVarTuple("Ts")

InAxes = int | None | Sequence[Any]
OutAxes = Any


def vmap(
    fun: Callable[P, R],
    in_axes: InAxes = 0,
    out_axes: OutAxes = 0,
    axis_name: Any = None,
    axis_size: int | None = None,
) -> Callable[P, R]:
    return jax.vmap(
        fun,
        in_axes=in_axes,
        out_axes=out_axes,
        axis_name=axis_name,
        axis_size=axis_size,
    )


def jit(
    fun: Callable[P, R],
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    donate_argnums: int | Sequence[int] | None = None,
    donate_argnames: str | Iterable[str] | None = None,
    device: JaxDevice | None = None,
) -> Callable[P, R]:
    return jax.jit(
        fun,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        donate_argnums=donate_argnums,
        donate_argnames=donate_argnames,
        device=device,
    )


def cond(
    pred: JArray,
    true_fun: Callable[..., TOut],
    false_fun: Callable[..., TOut],
    *operands: Any,
    **kwargs: Any,
) -> TOut:
    return lax.cond(pred, true_fun, false_fun, *operands, **kwargs)


def switch(
    index: JArray,
    branches: Sequence[Callable[P, R]],
    *operands: Any,
) -> R:
    return lax.switch(index, branches, *operands)


def fori_loop(
    lower: int,
    upper: int,
    body_fun: Callable[[int, T], T],
    init_val: T,
    *,
    unroll: int | bool | None = None,
) -> T:
    return lax.fori_loop(lower, upper, body_fun, init_val, unroll=unroll)


def while_loop(
    cond_fun: Callable[[T], JArray],
    body_fun: Callable[[T], T],
    init_val: T,
) -> T:
    return lax.while_loop(cond_fun, body_fun, init_val)


def scan(
    f: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X | None = None,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    **kwargs: Any,
) -> Tuple[Carry, Y]:
    return lax.scan(
        f, init, xs=xs, length=length, reverse=reverse, unroll=unroll, **kwargs
    )


@overload
def block_until_ready(x: T) -> T: ...
@overload
def block_until_ready(*args: Unpack[Ts]) -> tuple[Unpack[Ts]]: ...
def block_until_ready(*args):
    if len(args) == 1:
        return jax.block_until_ready(args[0])
    return tuple(jax.block_until_ready(x) for x in args)


def block_until_ready_all() -> None:
    """Best-effort barrier: block on all currently-live JAX arrays."""
    for a in jax.live_arrays():
        try:
            a.block_until_ready()
        except Exception:
            pass
