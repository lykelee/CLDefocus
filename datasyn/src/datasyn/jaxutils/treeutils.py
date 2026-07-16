import jax.tree as jtree

from datasyn.jaxutils.utils import *

T_PyTree = TypeVar("T_PyTree", bound=PyTree)


def tree_map(
    f: Callable[..., Any],
    tree: T_PyTree,
    *rest: Any,
    is_leaf: Callable[[T_PyTree], bool] | None = None,
) -> T_PyTree:
    return jtree.map(f, tree, *rest, is_leaf=is_leaf)


def tree_stack(tree: Sequence[T_PyTree], axis: int = 0) -> T_PyTree:
    """
    Given a sequence of pytrees with identical structure,
    returns a single pytree of the same structure where each
    leaf is `jnp.stack` of the corresponding leaves.
    """
    return jtree.map(lambda *leaves: jnp.stack(leaves, axis=axis), *tree)


def tree_batch_size(tree: T_PyTree, axis: int = 0) -> int:
    """
    Given a pytree whose leaves are JAX arrays all vectorized along `axis`,
    return the (common) size of that axis.

    Args:
      pytree:  any JAX-compatible pytree of arrays (e.g., dicts/lists of jnp.ndarray)
      axis:    which axis is the “batch” dimension (default: 0)

    Returns:
      An integer giving the size along that axis.

    Raises:
      ValueError: if no array-leaf is found, or if array-leaves disagree on that axis.
    """
    leaves = jtree.leaves(tree)
    batch_sizes = []
    for leaf in leaves:
        # skip non-array leaves (e.g., Python scalars, None, etc.)
        if hasattr(leaf, "shape"):
            try:
                batch_sizes.append(leaf.shape[axis])
            except IndexError:
                raise ValueError(f"Leaf with shape {leaf.shape} has no axis {axis}")
    if not batch_sizes:
        raise ValueError("No array leaves found in the pytree.")
    # verify consistency
    first = batch_sizes[0]
    for b in batch_sizes:
        if b != first:
            raise ValueError(f"Inconsistent batch sizes found: {batch_sizes}")
    return first


def tree_take(tree: T_PyTree, index: JArrayLike, axis: int | None = None) -> Any:
    return jtree.map(lambda x: jnp.take(x, index, axis=axis), tree)
