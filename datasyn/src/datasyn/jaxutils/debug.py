from typing import Type

import jax.lax as lax
import jax.numpy as jnp

from datasyn.jaxutils.typing import *

try:
    from jax.experimental import checkify as ck

    _HAS_CHECKIFY = True
except Exception:
    _HAS_CHECKIFY = False

_HAS_CALLBACK = hasattr(jax.debug, "callback")


def _py_raise(kind: Type[BaseException], msg: str) -> None:
    raise kind(msg)


def _reduce_bool(pred) -> BoolArray:
    if isinstance(pred, bool):
        return jnp.array(pred, dtype=bool)
    pred = jnp.asarray(pred, dtype=bool)
    return pred if pred.ndim == 0 else pred.reshape(-1).all()


def safe_assert(predicate, msg: str = "Assertion failed"):
    if isinstance(predicate, bool):
        if not predicate:
            raise AssertionError(msg)
        return

    bad = ~_reduce_bool(predicate)

    if _HAS_CALLBACK:

        def _fail(_):
            jax.debug.callback(_py_raise, AssertionError, msg)
            return ()

        def _ok(_):
            return ()

        lax.cond(bad, _fail, _ok, operand=None)
        return

    if _HAS_CHECKIFY:
        ck.assert_(~bad, msg)
        return

    def _trap(_):
        _ = 1.0 / jnp.array(0.0)
        return ()

    def _noop(_):
        return ()

    lax.cond(bad, _trap, _noop, operand=None)


def is_tracer(x: Any) -> bool:
    from jax.core import Tracer

    return isinstance(x, Tracer)


def is_traced(hint: Optional[JArray] = None):
    if hint is None:
        hint = jnp.empty(())
    return is_tracer(hint)
