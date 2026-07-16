from datasyn.jaxutils.utils import *


def flatten(x: JArray, start_axis: int = 0, end_axis: Optional[int] = None) -> JArray:
    return jax.lax.collapse(x, start_axis, end_axis)


def unflatten(x: JArray, axis: int, new_shape) -> JArray:
    pre_shape = x.shape[:axis]
    flat_dim = x.shape[axis]
    post_shape = x.shape[axis + 1 :]
    return x.reshape(pre_shape + tuple(new_shape) + post_shape)


def _deprecated_unsqueeze(x: JArray, axis: int | Sequence[int]):
    if isinstance(axis, int):
        axis = (axis,)
    axis = sorted(axis)
    new_shape = list(x.shape)
    for a in axis:
        new_shape.insert(a, 1)
    return jnp.reshape(x, tuple(new_shape))


def unsqueeze(x: JArray, axis: int | Sequence[int]):
    return jnp.expand_dims(x, axis)


def unsqueeze_and_repeat(x: JArray, axis: int, repeats: int):
    """
    Insert a new singleton axis at `axis`, then repeat along it `repeats` times.
    """
    # x has shape S = (s0, s1, ..., s{n-1})
    y = jnp.expand_dims(x, axis)  # shape = S[:axis] + (1,) + S[axis:]
    return jnp.repeat(y, repeats, axis=axis)


def slice_along_axis(x: JArray, slc: slice, axis: int):
    slicer = [slice(None)] * x.ndim
    slicer[axis] = slc
    return x[tuple(slicer)]


def chunk(
    x: JArray,
    chunk_size: int | None = None,
    chunk_num: int | None = None,
    axis: int = 0,
    inner_axis: Optional[int] = None,
    pad_value: Optional[Any] = None,
) -> JArray:
    """
    Split `x` along `axis` into blocks of size `chunk_size`, padding
    the final block so every chunk is full.
    """
    N = x.shape[axis]  # Axis size of chunk

    if chunk_size is None:
        if chunk_num is None:
            raise ValueError("One of chunk_size and chunk_num must be given!")

        # If axis size is smaller than chunk size, we reduce chunk size to the axis size.
        # This avoids unnecessary padding.
        # NOTE: This may break rule that the chunk axis size is always chunk_size!
        #       If this is a problem, we can add an option to control this behavior.
        if N < chunk_num:
            chunk_num = N

        pad_len = (-N) % chunk_num
        Np = x.shape[axis] + pad_len
        chunk_size = Np // chunk_num

    elif chunk_num is None:
        pad_len = (-N) % chunk_size
        Np = x.shape[axis] + pad_len
        chunk_num = Np // chunk_size

    else:
        # TODO: Pass if chunk size and num are consistent
        raise ValueError()

    if pad_len > 0:
        pad_width = [(0, 0)] * x.ndim
        pad_width[axis] = (0, pad_len)
        if pad_value is None:
            x = jnp.pad(x, pad_width, mode="edge")
            # pad_value = x.dtype.type(0)
        else:
            x = jnp.pad(x, pad_width, mode="constant", constant_values=pad_value)

    # reshape to (..., num_chunks, chunk_size, ...)
    shape = list(x.shape)
    shape[axis] = chunk_num
    shape.insert(axis + 1, chunk_size)
    x = x.reshape(shape)

    # move the within-chunk axis if the user asked for a custom position
    if inner_axis is None:
        inner_axis = axis + 1
    if inner_axis != axis + 1:
        x = jnp.moveaxis(x, axis + 1, inner_axis)

    return x


def unchunk(
    x_chk: JArray, chunk_axis: int = 0, inner_axis: int = 1, total: Optional[int] = None
):
    # TODO: Negative index handling!

    assert chunk_axis != inner_axis

    chk_num = x_chk.shape[chunk_axis]
    chk_size = x_chk.shape[inner_axis]
    chunk_total = chk_num * chk_size

    if total is None:
        total = chunk_total
    else:
        assert total > (chk_num - 1) * chk_size and total <= chunk_total, (
            "total size is wrong"
        )

    x_pad = flatten(jnp.moveaxis(x_chk, (chunk_axis, inner_axis), (0, 1)), 0, 2)
    x = slice_along_axis(x_pad, slice(0, total), axis=0)

    return x


def blockify(
    x: JArray,
    block_shape: Sequence[int] | None = None,
    block_num: Sequence[int] | None = None,
    axes: Optional[Sequence[int]] = None,
    *,
    pad_value: Optional[Any] = None,
) -> JArray:
    """
    Turn each axis in `axes` into two axes (n_blocks, block_size), padding the
    end so each original axis length is a multiple of its block size.

    Args:
      x:           input array.
      block_shape: shape of the blocks for each axis.
      axes:        which axes to block (defaults to 0, 1, ...).
    Kwargs:
      pad_value:   scalar pad, defaults to zero of x.dtype.

    Returns:
      An array whose first len(axes) dims are the number of blocks along
      each axis, then the next len(axes) dims are the block sizes themselves,
      then any remaining original axes.
    """
    if block_shape is not None:
        n_axes = len(block_shape)
    elif block_num is not None:
        n_axes = len(block_num)
    else:
        raise ValueError("One of `block_shape` and `block_num` should be given")

    if axes is None:
        axes = list(range(n_axes))
    if len(axes) != n_axes:
        raise ValueError("`axes` and `block_shape` must have same length")

    # work in ascending order of axis, tracking how many dims we've inserted
    axes_sorted = sorted(axes)
    offset = 0

    if block_shape is not None:
        for orig_ax, b in zip(axes_sorted, block_shape):
            ax = orig_ax + offset
            x = chunk(x, chunk_size=b, axis=ax, pad_value=pad_value)
            offset += 1  # each chunk adds one extra axis

    else:
        for orig_ax, b in zip(axes_sorted, block_num):
            ax = orig_ax + offset
            x = chunk(x, chunk_num=b, axis=ax, pad_value=pad_value)
            offset += 1  # each chunk adds one extra axis

    # after all chunk calls, the axes have been replaced in order by
    #   [n0, b0, n1, b1, ...], so now just transpose to [n0, n1, ..., b0, b1, ..., rest...]
    total_dims = x.ndim
    # indices of n's are even positions 0, 2, 4, ... up to 2*n_axes-2
    n_positions = list(range(0, 2 * n_axes, 2))
    # indices of b's are the odd positions 1, 3, ... up to 2*n_axes-1
    b_positions = list(range(1, 2 * n_axes, 2))
    rest = list(range(2 * n_axes, total_dims))
    perm = n_positions + b_positions + rest
    return jnp.transpose(x, perm)


def unblockify(
    x_blocks: JArray,
    ndim: int,
) -> JArray:
    """
    Merges block grid axes (0..ndim-1) and block axes (ndim..2*ndim-1)
    for each dimension, restoring the original array shape.
    This is a reverse of `blockify`.
    """
    # Compute the shape for merging
    shape = x_blocks.shape
    # For each axis, merge (n_blocks, block_size) -> (n_blocks * block_size)
    merged_shape = tuple(shape[i] * shape[ndim + i] for i in range(ndim))
    # Add any remaining axes
    rest_shape = shape[2 * ndim :]
    final_shape = merged_shape + rest_shape

    # Prepare permutation to interleave block grid and block axes
    # Current: [n0, n1, ..., b0, b1, ..., ...rest]
    # Want:    [n0, b0, n1, b1, ..., ...rest]
    perm = []
    for i in range(ndim):
        perm.append(i)
        perm.append(ndim + i)
    perm += list(range(2 * ndim, x_blocks.ndim))

    # Transpose to interleave, then reshape to merge
    x = jnp.transpose(x_blocks, perm)
    x = x.reshape(final_shape)
    return x
