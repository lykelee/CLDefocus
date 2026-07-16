"""
Point grid classes for systematic coordinate handling.

Two classes, two concepts:

  Grid         — infinite, indexed coordinate system. Pure geometry.
                 Both fields are JAX arrays; fully traceable inside jit/vmap.

  BoundedGrid  — finite region of a Grid, with a static shape.
                 Owns bounds logic; delegates coordinate math to Grid.
"""

from typing import Iterable

import chex
import jax
import jax.numpy as jnp

from datasyn.jaxutils.typing import *


@jax.tree_util.register_static
class StaticShape(tuple):
    """
    A static, hashable shape container for use in JAX pytrees.

    Registered with `@jax.tree_util.register_static` so JAX treats it as an
    opaque compile-time constant rather than tracing into its elements.
    Use this for shape fields in JAX-registered dataclasses.
    """

    __slots__ = ()

    def __new__(cls, values: Iterable[int]) -> "StaticShape":
        t = [int(x) for x in values]
        return tuple.__new__(cls, t)

    @property
    def ndim(self) -> int:
        return len(self)

    @property
    def size(self) -> int:
        result = 1
        for d in self:
            result *= d
        return result


@chex.dataclass(frozen=True)
class FixedSizeSlice:
    """
    A slice with a static size and a dynamic start position.

    Safe to pass through jit/vmap: start is traced, size is static.
    """

    start: IntArray
    size: StaticShape

    @classmethod
    def create(
        cls,
        start: IntArray | Sequence[int],
        size: tuple[int, ...],
    ) -> "FixedSizeSlice":
        return cls(start=jnp.asarray(start, dtype=jnp.int32), size=StaticShape(size))

    @classmethod
    def create_centered(
        cls,
        center: IntArray | Sequence[int],
        size: tuple[int, ...],
    ) -> "FixedSizeSlice":
        center = jnp.asarray(center, dtype=jnp.int32)
        size_arr = jnp.array(size, dtype=jnp.int32)
        return cls(start=center - size_arr // 2, size=StaticShape(size))

    @property
    def ndim(self) -> int:
        return self.size.ndim

    @property
    def end(self) -> IntArray:
        return self.start + jnp.array(self.size, dtype=jnp.int32)

    def dynamic_slice(self, x: JArray) -> JArray:
        return jax.lax.dynamic_slice(x, start_indices=self.start, slice_sizes=self.size)

    def dynamic_update_slice(self, x: JArray, update: JArray) -> JArray:
        return jax.lax.dynamic_update_slice(x, update, start_indices=self.start)

    def shift(self, offset: IntArray | Sequence[int]) -> "FixedSizeSlice":
        return FixedSizeSlice(
            start=self.start + jnp.asarray(offset, dtype=jnp.int32),
            size=self.size,
        )

    def clamp_to_array(self, array_shape: tuple[int, ...]) -> "FixedSizeSlice":
        max_start = jnp.array(array_shape, dtype=jnp.int32) - jnp.array(
            self.size, dtype=jnp.int32
        )
        return FixedSizeSlice(
            start=jnp.clip(self.start, 0, max_start),
            size=self.size,
        )

    def in_bounds(self, array_shape: tuple[int, ...]) -> BoolArray:
        shape_arr = jnp.array(array_shape, dtype=jnp.int32)
        return jnp.all(self.start >= 0) & jnp.all(self.end <= shape_arr)


class QuantizeResult(NamedTuple):
    """Result of a quantize (snap-to-grid) operation."""

    index: IntArray
    """Integer index of the nearest grid point. Shape (..., N)."""

    world: FloatArray
    """World coordinates of that nearest grid point. Shape (..., N)."""


@chex.dataclass(frozen=True)
class Grid:
    """
    An infinite, regularly-spaced, indexed coordinate system in N-D space.

    The grid point at integer index vector i is located at: origin + step * i.

    The sign of step encodes orientation:
      step[k] > 0  ->  index increases as the k-th coordinate increases.
      step[k] < 0  ->  index increases as the k-th coordinate decreases.

    Axis convention: the k-th element of origin/step corresponds to the k-th
    spatial axis (mathematical convention, not NumPy row-major convention).

    Both fields are JAX float arrays, so Grid is fully traceable inside
    jit/vmap. It carries no static (compile-time) data.

    Attributes:
        origin: World position of the grid point at index 0. Shape (N,).
        step:   Signed spacing per axis. Shape (N,).
                Magnitude = spacing; sign = orientation.
    """

    origin: FloatArray
    step: FloatArray

    # --- Basic properties ---

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return self.step.shape[0]

    @property
    def spacing(self) -> FloatArray:
        """Unsigned spacing between adjacent grid points. Shape (N,)."""
        return jnp.abs(self.step)

    @property
    def orientation(self) -> IntArray:
        """Sign of step per axis: +1 or -1. Shape (N,)."""
        return jnp.sign(self.step).astype(jnp.int32)

    # --- Coordinate transforms ---

    def index_to_world(self, index: IntArray) -> FloatArray:
        """
        Map an integer index vector to world coordinates.

        Args:
            index: Index vector. Shape (..., N).

        Returns:
            World coordinates. Shape (..., N).
        """
        return self.origin + self.step * index

    def world_to_index(self, point: FloatArray) -> FloatArray:
        """
        Map world coordinates to a continuous (non-integer) index.

        The result is not rounded; use quantize() to snap to the nearest
        grid point.

        Args:
            point: World coordinates. Shape (..., N).

        Returns:
            Continuous index. Shape (..., N).
        """
        return (point - self.origin) / self.step

    def quantize(self, point: FloatArray) -> QuantizeResult:
        """
        Snap a world point to the nearest grid point.

        Args:
            point: World coordinates. Shape (..., N).

        Returns:
            QuantizeResult:
              .index  — integer index of the nearest point. Shape (..., N).
              .world  — world coordinates of that point. Shape (..., N).
        """
        continuous = self.world_to_index(point)
        index = jnp.round(continuous).astype(jnp.int32)
        world = self.index_to_world(index)
        return QuantizeResult(index=index, world=world)

    # --- Grid transformations (return a new Grid) ---

    def flip(self, axes: int | tuple[int, ...] | None = None) -> "Grid":
        """
        Flip the orientation of the specified axes.

        Flipping axis k negates step[k], so that increasing index now moves
        in the opposite world direction. The origin is unchanged — index 0
        still maps to the same world point.

        Args:
            axes: Axis index, tuple of axis indices, or None to flip all axes.

        Returns:
            New Grid with flipped step on the specified axes.
        """
        if axes is None:
            return Grid(origin=self.origin, step=-self.step)
        if isinstance(axes, int):
            axes = (axes,)
        mask = jnp.ones(self.ndim, dtype=self.step.dtype)
        mask = mask.at[jnp.array(axes)].set(-1.0)
        return Grid(origin=self.origin, step=self.step * mask)

    def shift(self, delta_index: IntArray | tuple[int, ...]) -> "Grid":
        """
        Shift the origin so that what was at delta_index becomes the new index 0.

        After shifting, the point that was at index i is now at index i - delta_index.
        Equivalently, the world position of index 0 moves to the old index delta_index.

        delta_index may be a traced value (e.g., inside jit/vmap).

        Args:
            delta_index: Index offset. Shape (N,).

        Returns:
            New Grid with shifted origin, same step.
        """
        delta_index = jnp.asarray(delta_index)
        new_origin = self.origin + self.step * delta_index
        return Grid(origin=new_origin, step=self.step)

    def scale(self, s: float | FloatArray) -> "Grid":
        """
        Scale both step and origin by a scalar (or per-axis array).

        This scales the world coordinates of every grid point by s, preserving
        index assignments.

        Args:
            s: Scale factor. Scalar or Shape (N,).

        Returns:
            New Grid with step and origin both multiplied by s.
        """
        return Grid(origin=s * self.origin, step=s * self.step)

    def upsample(self, factor: int | tuple[int, ...]) -> "Grid":
        """
        Increase resolution by subdividing each step.

        The origin is unchanged. step[k] is divided by factor[k], so the
        same world region now has factor[k] times as many grid points along
        axis k.

        Args:
            factor: Subdivision factor per axis. Int for uniform, tuple for per-axis.

        Returns:
            New Grid with finer step, same origin.
        """
        if isinstance(factor, int):
            factor_arr = jnp.full(self.ndim, factor, dtype=self.step.dtype)
        else:
            factor_arr = jnp.asarray(factor, dtype=self.step.dtype)
        return Grid(origin=self.origin, step=self.step / factor_arr)

    def downsample(self, factor: int | tuple[int, ...]) -> "Grid":
        """
        Decrease resolution by merging steps.

        The origin is unchanged. step[k] is multiplied by factor[k].

        Args:
            factor: Decimation factor per axis. Int for uniform, tuple for per-axis.

        Returns:
            New Grid with coarser step, same origin.
        """
        if isinstance(factor, int):
            factor_arr = jnp.full(self.ndim, factor, dtype=self.step.dtype)
        else:
            factor_arr = jnp.asarray(factor, dtype=self.step.dtype)
        return Grid(origin=self.origin, step=self.step * factor_arr)

    # --- Conversion ---

    def bounded(self, shape: tuple[int, ...]) -> "BoundedGrid":
        """
        Attach a static shape to this grid, producing a BoundedGrid.

        Valid indices will be 0 <= i < shape[k] for each axis k.

        Args:
            shape: Number of grid points per axis (Python ints, static).

        Returns:
            BoundedGrid with this grid as its coordinate system.
        """
        return BoundedGrid(grid=self, shape=StaticShape(shape))


@chex.dataclass(frozen=True)
class BoundedGrid:
    """
    A finite region of a Grid, with a static shape.

    Valid index range: 0 <= i[k] < shape[k] for each axis k.
    Index 0 maps to grid.origin.

    The shape is always a StaticShape (a JAX-static tuple subclass). This
    is a hard invariant: shape must be known at compile time because it
    determines array sizes in indices() and grid_points(). For situations
    where the shape is unknown at compile time, use a plain Grid instead
    and pass the shape separately.

    Coordinate math is fully delegated to the underlying Grid. This class
    only owns bounds logic and shape-dependent operations.

    Attributes:
        grid:  The underlying infinite indexed coordinate system.
        shape: Number of grid points per axis. Always a static Python tuple.
    """

    grid: Grid
    shape: StaticShape

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    # --- Constructors ---

    @classmethod
    def stack(cls, grids: "Sequence[BoundedGrid]") -> "BoundedGrid":
        """
        Stack multiple 1-D BoundedGrids into a single N-D BoundedGrid.

        Each input grid must be exactly 1-dimensional, representing one spatial
        axis. The result has ndim = len(grids), with origin, step, and shape
        assembled in input order.

        This is the inverse of axis decomposition: if you have per-axis grids
        (e.g., from Bluestein CZT applied axis by axis), use this to reconstruct
        the joint N-D frequency grid.

        Args:
            grids: Sequence of 1-D BoundedGrids.

        Returns:
            N-D BoundedGrid with ndim = len(grids).

        Raises:
            ValueError: If any input grid is not 1-D.
        """
        for i, g in enumerate(grids):
            if g.ndim != 1:
                raise ValueError(
                    f"All grids must be 1-D, but grids[{i}] has ndim={g.ndim}"
                )
        new_origin = jnp.concatenate([g.origin for g in grids])
        new_step = jnp.concatenate([g.step for g in grids])
        new_shape = StaticShape(g.shape[0] for g in grids)
        return cls(grid=Grid(origin=new_origin, step=new_step), shape=new_shape)

    # --- Constructors ---

    @classmethod
    def from_origin_step(
        cls,
        origin: FloatArray,
        step: FloatArray,
        shape: tuple[int, ...],
    ) -> "BoundedGrid":
        """
        Construct directly from origin, step, and shape.

        Args:
            origin: World position of index 0. Shape (N,).
            step:   Signed spacing per axis. Shape (N,).
            shape:  Number of grid points per axis (Python ints).

        Returns:
            BoundedGrid.
        """
        return cls(
            grid=Grid(origin=jnp.asarray(origin), step=jnp.asarray(step)),
            shape=StaticShape(shape),
        )

    # --- Delegated properties ---

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return self.grid.ndim

    @property
    def origin(self) -> FloatArray:
        """World position of the grid point at index 0. Shape (N,)."""
        return self.grid.origin

    @property
    def step(self) -> FloatArray:
        """Signed spacing per axis. Shape (N,)."""
        return self.grid.step

    @property
    def spacing(self) -> FloatArray:
        """Unsigned spacing between adjacent grid points. Shape (N,)."""
        return self.grid.spacing

    @property
    def orientation(self) -> IntArray:
        """Sign of step per axis: +1 or -1. Shape (N,)."""
        return self.grid.orientation

    @property
    def size(self) -> int:
        """Total number of grid points (product of shape)."""
        return self.shape.size

    # --- Delegated coordinate transforms ---

    def index_to_world(self, index: IntArray) -> FloatArray:
        """
        Map an integer index vector to world coordinates.

        No bounds checking is performed.

        Args:
            index: Index vector. Shape (..., N).

        Returns:
            World coordinates. Shape (..., N).
        """
        return self.grid.index_to_world(index)

    def world_to_index(self, point: FloatArray) -> FloatArray:
        """
        Map world coordinates to a continuous (non-integer) index.

        Args:
            point: World coordinates. Shape (..., N).

        Returns:
            Continuous index. Shape (..., N).
        """
        return self.grid.world_to_index(point)

    def quantize(self, point: FloatArray) -> QuantizeResult:
        """
        Snap a world point to the nearest grid point.

        The returned index may lie outside the valid bounds.

        Args:
            point: World coordinates. Shape (..., N).

        Returns:
            QuantizeResult with .index and .world.
        """
        return self.grid.quantize(point)

    # --- Bounds operations ---

    def in_bounds(self, index: IntArray) -> BoolArray:
        """
        Check whether an index vector lies within [0, shape).

        Args:
            index: Index vector. Shape (..., N).

        Returns:
            Boolean. Shape (...,).
        """
        index = jnp.asarray(index)
        shape_arr = jnp.array(self.shape, dtype=jnp.int32)
        return jnp.all((index >= 0) & (index < shape_arr), axis=-1)

    def clamp_index(self, index: IntArray) -> IntArray:
        """
        Clamp an index vector to the valid range [0, shape - 1].

        Args:
            index: Index vector. Shape (..., N).

        Returns:
            Clamped index. Shape (..., N).
        """
        index = jnp.asarray(index)
        shape_arr = jnp.array(self.shape, dtype=jnp.int32)
        return jnp.clip(index, 0, shape_arr - 1)

    # --- Enumeration ---

    def indices(self) -> IntArray:
        """
        Generate all valid index vectors.

        Returns:
            Array of shape (*shape, N) containing every index vector in the grid.
            Axis ordering matches the mathematical convention: indices()[i, j, k]
            gives the index vector (i, j, k) as a length-N array.
        """
        ranges = [jnp.arange(s, dtype=jnp.int32) for s in self.shape]
        grids = jnp.meshgrid(*ranges, indexing="ij")
        return jnp.stack(grids, axis=-1)

    def grid_points(self) -> FloatArray:
        """
        Generate world coordinates of all grid points.

        Returns:
            Array of shape (*shape, N).
        """
        return self.index_to_world(self.indices())

    # --- Corner / range queries ---

    @property
    def first_index(self) -> IntArray:
        """First valid index vector: (0, 0, ..., 0). Shape (N,)."""
        return jnp.zeros(self.ndim, dtype=jnp.int32)

    @property
    def last_index(self) -> IntArray:
        """Last valid index vector (inclusive): (shape[0]-1, ...). Shape (N,)."""
        return jnp.array(self.shape, dtype=jnp.int32) - 1

    @property
    def end_index(self) -> IntArray:
        """One-past-last index vector (exclusive): shape. Shape (N,).

        Use this for numpy-style slicing: arr[s:e] where e = end_index.
        """
        return jnp.array(self.shape, dtype=jnp.int32)

    def corner_points(self) -> tuple[FloatArray, FloatArray]:
        """
        Return the world coordinates of the first and last grid points (inclusive).

        Returns:
            (first_point, last_point), both shape (N,).
        """
        return self.index_to_world(self.first_index), self.index_to_world(
            self.last_index
        )

    def world_range(self) -> tuple[FloatArray, FloatArray]:
        """
        Return the world coordinates of the first point (inclusive) and the
        one-past-last point (exclusive).

        This gives the half-open interval [start, end) in world space, which
        matches the convention used by range/CZT arguments (e.g., Bluestein
        f1/f2). The end point is the world position of end_index (= shape),
        one step past the last valid grid point.

        Returns:
            (start, end), both shape (N,).
            start = world coordinate of index 0        (= origin)
            end   = world coordinate of index shape    (exclusive)
        """
        return self.origin, self.index_to_world(self.end_index)

    # --- Transformations (return a new BoundedGrid) ---

    def slice(
        self,
        start: IntArray | tuple[int, ...],
        shape: tuple[int, ...],
    ) -> "BoundedGrid":
        """
        Extract a sub-region starting at start with the given shape.

        The returned BoundedGrid has index 0 at the world position of start
        in this grid. start may be a traced value (dynamic inside jit/vmap).
        shape must be a static tuple of Python ints.

        Args:
            start: Starting index in this grid. Shape (N,). May be traced.
            shape: Number of grid points per axis in the sub-region (static).

        Returns:
            New BoundedGrid covering the sub-region.
        """
        return BoundedGrid(grid=self.grid.shift(start), shape=StaticShape(shape))

    def with_shape(self, shape: tuple[int, ...]) -> "BoundedGrid":
        """
        Return a new BoundedGrid with a different shape but the same origin.

        Args:
            shape: New shape (static).

        Returns:
            New BoundedGrid.
        """
        return BoundedGrid(grid=self.grid, shape=StaticShape(shape))

    def scale(self, s: "float | FloatArray") -> "BoundedGrid":
        """
        Scale both step and origin by a scalar (or per-axis array).

        The shape is unchanged — the same number of grid points now covers a
        different world range. Use this to convert between coordinate systems
        (e.g., spatial to frequency via f_scale = 1 / (wvl * z)).

        Args:
            s: Scale factor. Scalar or Shape (N,).

        Returns:
            New BoundedGrid with step and origin both multiplied by s.
        """
        return BoundedGrid(grid=self.grid.scale(s), shape=self.shape)

    def flip(self, axes: int | tuple[int, ...] | None = None) -> "BoundedGrid":
        """
        Flip the orientation of the specified axes.

        Index 0 still maps to the same world point; increasing indices now
        move in the opposite direction.

        Args:
            axes: Axis or axes to flip. None flips all.

        Returns:
            New BoundedGrid with flipped orientation, same shape.
        """
        return BoundedGrid(grid=self.grid.flip(axes), shape=self.shape)

    def upsample(self, factor: int | tuple[int, ...]) -> "BoundedGrid":
        """
        Increase resolution by subdividing each step.

        World coverage is unchanged; shape is multiplied by factor.

        Args:
            factor: Subdivision factor. Int for uniform, tuple for per-axis.

        Returns:
            New BoundedGrid with finer grid and larger shape.
        """
        if isinstance(factor, int):
            new_shape = StaticShape(s * factor for s in self.shape)
        else:
            new_shape = StaticShape(s * f for s, f in zip(self.shape, factor))
        return BoundedGrid(grid=self.grid.upsample(factor), shape=new_shape)

    def downsample(self, factor: int | tuple[int, ...]) -> "BoundedGrid":
        """
        Decrease resolution by merging steps.

        World coverage is unchanged; shape is divided by factor (integer div).

        Args:
            factor: Decimation factor. Int for uniform, tuple for per-axis.

        Returns:
            New BoundedGrid with coarser grid and smaller shape.
        """
        if isinstance(factor, int):
            new_shape = StaticShape(s // factor for s in self.shape)
        else:
            new_shape = StaticShape(s // f for s, f in zip(self.shape, factor))
        return BoundedGrid(grid=self.grid.downsample(factor), shape=new_shape)
