from dataclasses import dataclass
from typing import Optional, Self, Sequence, Union

import jax
import jax.numpy as jnp

import datasyn.optics.safeop as safeop
from datasyn.mathutils.vecop import normdir
from datasyn.optics.typing import *

T = TypeVar("T")


def try_reshape(x: T, shape) -> T:
    if hasattr(x, "reshape"):
        return x.reshape(shape)
    return x


def try_getitem(x: T, idx) -> T:
    if hasattr(x, "__getitem__"):
        return x[idx]
    return x


class RayBase:
    """
    An interface for general rays.
    `RayGeo` only has geometrical information o and d.
    Other rays may have other information (e.g., optical path length for optics).

    TODO: Can we deal with infinite origins with homogeneous coordiantes?
    """

    @property
    def o(self) -> JArray:
        raise NotImplementedError()

    @property
    def d(self) -> JArray:
        raise NotImplementedError()

    def set_o(self, o: JArray) -> Self:
        raise NotImplementedError()

    def set_d(self, d: JArray) -> Self:
        raise NotImplementedError()

    def propagate(self, t: Union[float, JArray], **kwargs) -> Self:
        raise NotImplementedError()

    def propagate_o(self, t: Union[float, JArray]) -> JArray:
        raise NotImplementedError()

    @property
    def shape(self) -> Tuple[int, ...]:
        raise NotImplementedError()

    @property
    def ndim(self) -> Tuple[int, ...]:
        raise NotImplementedError()

    @property
    def dim(self) -> int:
        """Dimension of space (e.g., 3 for rays in 3D space)"""
        raise NotImplementedError()


TRay = TypeVar("TRay", bound=RayBase)


class ProjectToZOut(NamedTuple, Generic[TRay]):
    t: FloatArray
    ray: TRay


def project_to_z(z: JArray, ray: TRay):
    z = jnp.asarray(z)
    if ray.ndim != 0 or z.shape != ():
        raise NotImplementedError()

    t = (z - ray.o[..., 2]) / ray.d[..., 2]
    ray_ = ray.propagate(t)
    return ProjectToZOut(t=t, ray=ray_)


def opposite(ray: TRay):
    """
    Flips the ray direction i.e. (o, d) -> (o, -d).
    """
    return ray.set_d(-ray.d)


def translate(ray: TRay, do: JArray):
    """
    Translates the ray origin (with no propagation).
    """
    return ray.set_o(ray.o + do)


def _canonicalize_batch_index(n_batch: int, idx: Any) -> Tuple[Any, ...]:
    """
    Build an index tuple that targets exactly the batch axes (length == n_batch consuming indices),
    while preserving any inserted newaxes (None). Ellipsis expands only over batch axes.
    """
    if not isinstance(idx, tuple):
        idx = (idx,)

    # Validate ellipsis count
    if sum(x is Ellipsis for x in idx) > 1:
        raise IndexError("an index can only have a single ellipsis")

    # Count how many entries *consume* batch axes (everything except None/Ellipsis)
    def consumes(x: Any) -> bool:
        return (x is not None) and (x is not Ellipsis)

    consuming = sum(consumes(x) for x in idx)

    if consuming > n_batch:
        raise IndexError(
            f"too many indices for batch axes: got {consuming}, but only {n_batch} batch axes"
        )

    # Expand ellipsis into the right number of ":" to cover remaining batch axes
    if Ellipsis in idx:
        e = idx.index(Ellipsis)
        before, after = idx[:e], idx[e + 1 :]

        consuming_before = sum(consumes(x) for x in before)
        consuming_after = sum(consumes(x) for x in after)
        needed = n_batch - (consuming_before + consuming_after)

        idx = before + (slice(None),) * needed + after

    # If still short, pad with ":" at the end (these consume remaining batch axes)
    consuming2 = sum(consumes(x) for x in idx)
    idx = idx + (slice(None),) * (n_batch - consuming2)

    return idx


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class RayGeo(RayBase):
    """
    A data structure for a ray geometry (origin and direction).
    """

    _o: JArray
    _d: JArray

    @property
    def o(self):
        return self._o

    @property
    def d(self):
        return self._d

    def set_o(self, o: JArray):
        return RayGeo(o, self._d)

    def set_d(self, d: JArray):
        return RayGeo(self._o, d)

    def propagate(self, t: Union[float, JArray], **kwargs):
        o_ = self._o + t * self._d
        return RayGeo(o_, self._d)

    def propagate_o(self, t: Union[float, JArray]):
        return self._o + t * self._d

    @property
    def od(self):
        return jnp.concat([self._o, self._d], axis=-1)

    @property
    def do(self):
        return jnp.concat([self._d, self._o], axis=-1)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._o.shape[:-1]

    @property
    def ndim(self) -> Tuple[int, ...]:
        return self._o.ndim - 1

    @property
    def dim(self) -> int:
        """Dimension of space (e.g., 3 for rays in 3D space)"""
        return self._o.shape[-1]

    def reshape(self, new_shape: Sequence[int]):
        d = self.dim
        return RayGeo(
            self._o.reshape(tuple(new_shape) + (d,)),
            self._d.reshape(tuple(new_shape) + (d,)),
        )

    def __getitem__(self, idx):
        idx = _canonicalize_batch_index(self.ndim, idx)
        return RayGeo(self._o[idx], self._d[idx])

    @staticmethod
    def zeros(shape=(), dim=3):
        """
        Rays with the zero origin and the +z direction.
        """
        return RayGeo(
            jnp.zeros(shape + (dim,)),
            jnp.concat(
                [jnp.zeros(shape + (dim - 1,)), jnp.ones(shape + (1,))], axis=-1
            ),
        )


def generate_origin_direction_pairs(
    origins: JArray,
    targets: JArray,
    unitdir: bool = True,
) -> Tuple[JArray, JArray]:
    """
    Generate all origin-direction pairs for arbitrary batch axes.

    Given origins of shape (*o, dim) and targets of shape (*t, dim),
    this function returns:
      - origins_out of shape (*o, *t, dim)
      - directions of shape (*o, *t, dim) with normalized direction vectors
        from each origin to each target.

    Args:
        origins: JArray of shape (*o, dim) representing origin points.
        targets: JArray of shape (*t, dim) representing target points.

    Returns:
        A tuple of two JArrays (origins_out, directions) both of shape (*o, *t, dim).
    """
    dim = origins.shape[-1]
    assert targets.shape[-1] == dim

    # Extract batch dimensions (all but last axis)
    orig_shape = origins.shape[:-1]  # shape *o
    tgt_shape = targets.shape[:-1]  # shape *t

    # Determine the number of batch dimensions
    orig_ndim = len(orig_shape)
    tgt_ndim = len(tgt_shape)

    # Expand dimensions to enable broadcasting over combined batch axes
    origins_expanded = origins.reshape(orig_shape + (1,) * tgt_ndim + (dim,))
    targets_expanded = targets.reshape((1,) * orig_ndim + tgt_shape + (dim,))

    # Compute differences: shape becomes (*o, *t, dim)
    diff = targets_expanded - origins_expanded
    dirs = normdir(diff) if unitdir else diff

    # Broadcast origins over target batch axes to shape (*o, *t, dim)
    final_shape = orig_shape + tgt_shape + (dim,)
    origins_out = jnp.broadcast_to(origins_expanded, final_shape)

    return origins_out, dirs


def generate_rays_from_origin_target(origins: JArray, targets: JArray):
    o, d = generate_origin_direction_pairs(origins, targets, unitdir=True)
    return RayGeo(_o=o, _d=d)


DEFAULT_WAVELENGTH_NM = 587.6  # Fraunhofer line d
DEFAULT_WAVELENGTH = DEFAULT_WAVELENGTH_NM * 1e-9


class LensRay(RayBase):
    """A lens ray: blockable, medium-aware (ior), and wavelength-carrying."""

    @property
    def valid(self):
        raise NotImplementedError

    def block(self, mask: BoolArray) -> "LensRay":
        raise NotImplementedError

    def propagate(self, t: Union[float, JArray], *, ior: float = 1.0):
        raise NotImplementedError

    @property
    def wvln(self) -> FloatArray:
        raise NotImplementedError


TLensRay = TypeVar("TLensRay", bound=LensRay)


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class ORay(LensRay):
    g: RayGeo
    _wvln: FloatArray = None
    _opl: Optional[FloatArray] = None
    _valid: Optional[BoolArray] = None
    _pss: Optional[FloatArray] = None

    @property
    def shape(self):
        return self.g.shape

    @property
    def ndim(self):
        return self.g.ndim

    @property
    def dim(self):
        return self.g.dim

    @property
    def o(self):
        return self.g._o

    @property
    def d(self):
        return self.g._d

    @property
    def wvln(self):
        return self._wvln

    @property
    def valid(self):
        if self._valid is None:
            return jnp.ones(self.shape, dtype=bool)
        return self._valid

    @property
    def amp(self):
        if self._valid is None:
            return jnp.ones(self.shape, dtype=float)
        return 1.0 * self._valid

    @property
    def opl(self):
        if self._opl is None:
            return jnp.zeros_like(self.o[..., 0])
        return self._opl

    def block(self, mask: BoolArray) -> "ORay":
        return self.set_valid(self.valid & mask)

    def set_g(self, g: RayGeo):
        return ORay(
            g=g, _wvln=self._wvln, _opl=self._opl, _valid=self._valid, _pss=self._pss
        )

    def set_o(self, o: JArray):
        return self.set_g(self.g.set_o(o))

    def set_d(self, d: JArray):
        return self.set_g(self.g.set_d(d))

    def set_opl(self, opl: JArray):
        if self._opl is None:
            return self
        return ORay(
            g=self.g, _wvln=self._wvln, _opl=opl, _valid=self._valid, _pss=self._pss
        )

    def set_valid(self, valid: BoolArray):
        if self._valid is None:
            return self
        return ORay(
            g=self.g, _wvln=self._wvln, _opl=self._opl, _valid=valid, _pss=self._pss
        )

    def set_pss(self, pss: FloatArray):
        if self._pss is None:
            return self
        return ORay(
            g=self.g, _wvln=self._wvln, _opl=self._opl, _valid=self.valid, _pss=pss
        )

    def reshape(self, new_shape: Sequence[int]):
        return ORay(
            g=self.g.reshape(new_shape),
            _wvln=try_reshape(self._wvln, new_shape),
            _opl=try_reshape(self._opl, new_shape),
            _valid=try_reshape(self._valid, new_shape),
            _pss=try_reshape(self._pss, new_shape),
        )

    def __getitem__(self, idx):
        return ORay(
            g=self.g[idx],
            _wvln=try_getitem(self._wvln, idx),
            _opl=try_getitem(self._opl, idx),
            _valid=try_getitem(self._valid, idx),
            _pss=try_getitem(self._pss, idx),
        )

    def propagate(self, t: Union[float, JArray], *, ior: float = 1.0):
        ray = self.set_g(self.g.propagate(t[..., None]))
        if self._opl is not None:
            opl = self._opl + (ior * t)
            ray = ray.set_opl(opl)
        return ray

    def propagate_o(self, t: Union[float, JArray]):
        return self.o + t[..., None] * self.d


def make_oray(
    *,
    g: Optional[RayGeo] = None,
    o: Optional[FloatArray] = None,
    d: Optional[FloatArray] = None,
    wvln: float | FloatArray = DEFAULT_WAVELENGTH,
    opl: Optional[float | FloatArray] = None,
    valid: Optional[bool | BoolArray] = None,
    pss: Optional[float | FloatArray] = None,
):
    if g is None:
        if o is None or d is None:
            raise ValueError
        d = safeop.normdir(d).v
        o, d = jnp.broadcast_arrays(o, d)
        g = RayGeo(o, d)

    wvln = jnp.asarray(wvln)
    if opl is not None:
        opl = jnp.broadcast_to(jnp.asarray(opl), g.shape)
    if valid is not None:
        valid = jnp.broadcast_to(jnp.asarray(valid), g.shape)
    if pss is not None:
        pss = jnp.broadcast_to(jnp.asarray(pss), g.shape)

    return ORay(g=g, _wvln=wvln, _opl=opl, _valid=valid, _pss=pss)


def oray_intersection(rays: ORay):
    from datasyn.mathutils.intersection import best_fit_line_intersection

    p, _, _ = best_fit_line_intersection(rays.o, rays.d, rays.amp)
    return p
