from functools import partial

import jax.lax as lax

import datasyn.jaxutils.nputils as nputils
import datasyn.jaxutils.treeutils as treeutils
import datasyn.jaxutils.typevars as tpvars
from datasyn.jaxutils.utils import *

# ---------------------------------------------------------------------------
# vmap helpers
# ---------------------------------------------------------------------------

Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")


class ScanCallback(Generic[Carry, Y]):
    __call__: Callable[[int, Carry, Y], None]


def fake_scan(
    f: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X | None = None,
    length: int | None = None,
    reverse: bool = False,
    loop_wrapper: Optional[Callable[[Iterable], Iterable]] = None,
    callback: Optional[ScanCallback[Carry, Y]] = None,
) -> Tuple[Carry, Y]:
    """
    A python-only version of `jax.lax.scan` for debugging.
    """
    if xs is not None:
        N = treeutils.tree_batch_size(xs, axis=0)
    else:
        if length is None:
            raise ValueError("fake_scan: must specify length when xs is None")
        N = length

    idxs = list(range(N))
    if reverse:
        idxs = idxs[::-1]

    carry = init

    if xs is None:
        for i in idxs:
            carry, y_i = f(carry, None)
            if callback is not None:
                callback(i, carry, y_i)
        return carry, None

    ys_list = []

    iters = idxs
    if loop_wrapper is not None:
        iters = loop_wrapper(iters)

    for i in iters:
        x_i = treeutils.tree_take(xs, index=i, axis=0)
        carry, y_i = f(carry, x_i)
        ys_list.append(y_i)
        if callback is not None:
            callback(i, carry, y_i)

    ys = treeutils.tree_stack(ys_list, axis=0)

    return carry, ys


def chunk_vmapped(
    fun_chunked: Callable[tpvars.P, tpvars.R],
    chunk_size: int | None,
    disable_scan_trace: bool = False,
) -> Callable[tpvars.P, tpvars.R]:
    """
    Run an already-vectorized ``fun_chunked`` over the leading batch axis in
    chunks of ``chunk_size`` to bound peak memory. Chunks are iterated with
    ``lax.scan`` (or the python ``fake_scan`` when ``disable_scan_trace``).
    """

    def fn(*args):
        total = treeutils.tree_batch_size(args, axis=0)
        if (chunk_size is None) or (total <= chunk_size):
            return fun_chunked(*args)

        fn_chunk = partial(nputils.chunk, chunk_size=chunk_size)
        args_chk = treeutils.tree_map(fn_chunk, args)

        def body(_, packed_chunk):
            out_chunk = fun_chunked(*packed_chunk)
            return None, out_chunk

        scan_fn = fake_scan if disable_scan_trace else lax.scan
        _, ret_chk = scan_fn(body, None, args_chk)
        y = treeutils.tree_map(lambda x: nputils.unchunk(x, total=total), ret_chk)

        return y

    return fn
