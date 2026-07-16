import chex
import jax.numpy as jnp

from datasyn.jaxutils.typing import *
from datasyn.mathutils.grid import BoundedGrid, StaticShape


def _default_offset(ndim: int) -> tuple[int, ...]:
    """Default offset (0, 0, ...) for DFT arrangement."""
    return tuple(0 for _ in range(ndim))


def _centered_offset(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Centered offset (-N//2, ...) for each axis."""
    return tuple(-(n // 2) for n in shape)


@chex.dataclass(frozen=True)
class DFTSampler:
    """
    DFT sampling specification: spacing, count, and offset per axis.

    Defines sample positions:
        x_i = (i + offset[axis]) * spacing[axis]  for i = 0, ..., shape[axis]-1

    The offset tracks where samples are physically located:
    - offset = (0, 0, ...): DFT arrangement, x_i = i * dx (native FFT layout)
    - offset = (-N//2, ...): Centered arrangement, x_i = (i - N/2) * dx

    Both shape and offset are static Python tuples (never JAX-traced).
    Always construct via create() or create_centered() rather than directly,
    to ensure proper type wrapping.

    Attributes:
        spacing: Positive spacing per axis. Shape (ndim,).
        shape:   Sample count per axis (static, not traced).
        offset:  Integer offset per axis (static, not traced).
    """

    spacing: FloatArray
    shape: StaticShape
    offset: tuple[int, ...]

    @classmethod
    def create(
        cls,
        spacing: FloatArray | Sequence[float],
        shape: tuple[int, ...] | Sequence[int],
        offset: tuple[int, ...] | Sequence[int] | None = None,
    ) -> "DFTSampler":
        """
        Create a DFTSampler.

        Args:
            spacing: Positive spacing per axis.
            shape:   Sample count per axis.
            offset:  Integer offset per axis. Default (0, 0, ...) for DFT arrangement.

        Returns:
            A DFTSampler instance.
        """
        spacing = jnp.asarray(spacing)
        shape = StaticShape(shape)
        if len(shape) != spacing.shape[0]:
            raise ValueError(
                f"spacing length {spacing.shape[0]} != shape length {len(shape)}"
            )
        if offset is None:
            offset = _default_offset(len(shape))
        else:
            offset = tuple(int(o) for o in offset)
            if len(offset) != len(shape):
                raise ValueError(
                    f"offset length {len(offset)} != shape length {len(shape)}"
                )
        return cls(spacing=spacing, shape=shape, offset=offset)

    @classmethod
    def create_centered(
        cls,
        spacing: FloatArray | Sequence[float],
        shape: tuple[int, ...] | Sequence[int],
    ) -> "DFTSampler":
        """
        Create a centered DFTSampler (offset = -N//2 per axis).

        Args:
            spacing: Positive spacing per axis.
            shape:   Sample count per axis.

        Returns:
            A DFTSampler with centered offset.
        """
        shape = StaticShape(shape)
        return cls.create(spacing=spacing, shape=shape, offset=_centered_offset(shape))

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Total number of sample points."""
        return self.shape.size

    @property
    def extent(self) -> FloatArray:
        """
        Total extent per axis: shape * spacing.

        This is the period of the DFT in each dimension.
        """
        return jnp.asarray(self.shape) * self.spacing

    @property
    def is_dft_arrangement(self) -> bool:
        """True if offset is all zeros (native DFT layout)."""
        return all(o == 0 for o in self.offset)

    @property
    def is_centered(self) -> bool:
        """True if offset equals centered offset (-N//2, ...)."""
        return self.offset == _centered_offset(self.shape)

    def positions_1d(self, axis: int) -> FloatArray:
        """
        Get sample positions along one axis.

        Args:
            axis: Which axis.

        Returns:
            Positions [(0 + offset)*dx, (1 + offset)*dx, ..., (N-1 + offset)*dx].
        """
        return (jnp.arange(self.shape[axis]) + self.offset[axis]) * self.spacing[axis]

    def positions(self) -> FloatArray:
        """
        Get all sample positions as a meshgrid.

        Returns:
            Array of shape (*shape, ndim) containing position vectors.
        """
        axes = [self.positions_1d(ax) for ax in range(self.ndim)]
        grids = jnp.meshgrid(*axes, indexing="ij")
        return jnp.stack(grids, axis=-1)

    def with_offset(self, new_offset: tuple[int, ...] | Sequence[int]) -> "DFTSampler":
        """
        Create a new sampler with a different offset (metadata only, no rolling).

        NOTE: Use DFTSampleGrid.with_offset() to also roll the values.
        """
        return DFTSampler(
            spacing=self.spacing,
            shape=self.shape,
            offset=tuple(int(o) for o in new_offset),
        )

    def to_dft_sampler(self) -> "DFTSampler":
        """Create a sampler with DFT arrangement (offset = 0)."""
        return self.with_offset(_default_offset(self.ndim))

    def to_centered_sampler(self) -> "DFTSampler":
        """Create a sampler with centered arrangement (offset = -N//2)."""
        return self.with_offset(_centered_offset(self.shape))

    @property
    def frequency_spacing(self) -> FloatArray:
        """
        Frequency spacing corresponding to this spatial sampler.

        df = 1 / (N * dx) = 1 / extent
        """
        return 1.0 / self.extent

    @property
    def frequency_sampler(self) -> "DFTSampler":
        """
        Corresponding frequency domain sampler.

        The frequency sampler has:
        - spacing: df = 1 / (N * dx)
        - shape: same as spatial shape
        - offset: same as spatial offset (preserves arrangement)
        """
        return DFTSampler(
            spacing=self.frequency_spacing, shape=self.shape, offset=self.offset
        )

    def to_spatial(self) -> "DFTSampler":
        """
        Interpret this as a frequency sampler and return the corresponding spatial sampler.

        If this sampler has spacing df, the spatial sampler has dx = 1 / (N * df).
        Preserves the offset.
        """
        return DFTSampler(
            spacing=1.0 / self.extent, shape=self.shape, offset=self.offset
        )

    def to_grid(self) -> BoundedGrid:
        """Convert to a BoundedGrid."""
        return BoundedGrid.from_origin_step(
            origin=jnp.asarray(self.offset, dtype=self.spacing.dtype) * self.spacing,
            step=self.spacing,
            shape=tuple(self.shape),
        )


@chex.dataclass
class DFTSampleGrid:
    """
    DFT samples: sampler geometry + value array.

    The sampler's offset tracks where samples are physically located.
    Use to_dft_arrangement() before FFT if offset != 0.

    Attributes:
        sampler: The sampling specification (includes offset).
        values:  Sample values. Shape must be compatible with sampler.shape.
    """

    sampler: DFTSampler
    values: JArray

    @classmethod
    def create(
        cls,
        spacing: FloatArray | Sequence[float],
        shape: tuple[int, ...] | Sequence[int],
        values: JArray,
        offset: tuple[int, ...] | Sequence[int] | None = None,
    ) -> "DFTSampleGrid":
        """
        Create a DFTSampleGrid.

        Args:
            spacing: Positive spacing per axis.
            shape:   Sample count per axis.
            values:  Sample values.
            offset:  Integer offset per axis. Default (0, ...) for DFT arrangement.
        """
        sampler = DFTSampler.create(spacing=spacing, shape=shape, offset=offset)
        return cls(sampler=sampler, values=values)

    # --- Shortcut properties ---

    @property
    def spacing(self) -> FloatArray:
        """Spacing per axis."""
        return self.sampler.spacing

    @property
    def shape(self) -> StaticShape:
        """Sample shape."""
        return self.sampler.shape

    @property
    def offset(self) -> tuple[int, ...]:
        """Offset per axis."""
        return self.sampler.offset

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.sampler.ndim

    @property
    def extent(self) -> FloatArray:
        """Total extent per axis: shape * spacing."""
        return self.sampler.extent

    @property
    def is_dft_arrangement(self) -> bool:
        """True if in DFT arrangement (offset = 0)."""
        return self.sampler.is_dft_arrangement

    def positions(self) -> FloatArray:
        """Get all sample positions."""
        return self.sampler.positions()

    # --- Functional updates ---

    def with_values(self, new_values: JArray) -> "DFTSampleGrid":
        """New grid with different values, same sampler."""
        return DFTSampleGrid(sampler=self.sampler, values=new_values)

    def with_sampler(self, new_sampler: DFTSampler) -> "DFTSampleGrid":
        """New grid with different sampler, same values."""
        return DFTSampleGrid(sampler=new_sampler, values=self.values)

    # --- Offset / arrangement conversion ---

    def _roll_to_offset(
        self, new_offset: tuple[int, ...], axis=None
    ) -> "DFTSampleGrid":
        """
        Roll values to match a new offset.

        Args:
            new_offset: Target offset per axis.
            axis: If None, roll all axes. If int or tuple of ints, only roll
                  those axes (used by bluestein for partial rolling).
        """
        # NOTE: Be careful of JAX tracer issue!

        full_shift = tuple(new - cur for new, cur in zip(new_offset, self.offset))

        if axis is None:
            axes = tuple(range(self.ndim))
        elif isinstance(axis, int):
            axes = (axis,)
        else:
            axes = tuple(axis)

        shifts = tuple(full_shift[a] for a in axes)
        rolled = jnp.roll(self.values, shift=shifts, axis=axes)

        return DFTSampleGrid(
            sampler=self.sampler.with_offset(new_offset), values=rolled
        )

    def to_dft_arrangement(self, axis=None) -> "DFTSampleGrid":
        """
        Convert to DFT arrangement (offset = 0) by rolling values.

        If already in DFT arrangement, returns self unchanged.
        """
        return self._roll_to_offset(_default_offset(self.ndim), axis=axis)

    def to_centered(self, axis=None) -> "DFTSampleGrid":
        """
        Convert to centered arrangement (offset = -N//2) by rolling values.

        Args:
            axis: If None, roll all axes. If specified, only roll that axis.
                  (Partial rolling used by bluestein-based transforms.)
        """
        return self._roll_to_offset(_centered_offset(self.shape), axis=axis)

    def extents(self) -> FloatArray:
        """Total extent per axis: shape * spacing. Alias for the extent property."""
        return self.extent

    def with_offset(
        self, new_offset: tuple[int, ...] | Sequence[int]
    ) -> "DFTSampleGrid":
        """
        Change to an arbitrary offset arrangement by rolling values.

        Args:
            new_offset: New offset per axis.
        """
        new_offset = tuple(int(o) for o in new_offset)
        if self.offset == new_offset:
            return self
        return self._roll_to_offset(new_offset)
