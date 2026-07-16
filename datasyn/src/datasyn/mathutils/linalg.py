import jax.numpy as jnp

from datasyn.jaxutils.typing import *


def masked_lstsq(A: JArray, b: JArray, mask: BoolArray, rcond: float | None = None):
    """0/1-weighted least square; values at masked entries are meaningless."""
    w = mask.astype(A.dtype)
    Aw = A * w[:, None]
    bw = b * w
    return jnp.linalg.lstsq(Aw, bw, rcond=rcond)


def masked_lstsq_lineax(A: JArray, b: JArray, mask: bool | BoolArray):
    """0/1-weighted least square via lineax; values at masked entries are meaningless."""
    import lineax as lx

    w = jnp.asarray(mask).astype(A.dtype)
    w = jnp.broadcast_to(w, b.shape)
    Aw = A * w[:, None]
    bw = b * w

    operator = lx.MatrixLinearOperator(Aw)
    solution = lx.linear_solve(
        operator, bw, solver=lx.AutoLinearSolver(well_posed=None)
    )

    return solution.value
