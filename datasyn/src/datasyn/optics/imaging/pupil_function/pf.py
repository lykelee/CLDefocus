from dataclasses import dataclass

import jax.numpy as jnp

import datasyn.mathutils.vecop as vecop
import datasyn.optics.ray as ray
import datasyn.optics.safeop as safeop
from datasyn.jaxutils import debug as debugjax
from datasyn.jaxutils import jx, wrappings
from datasyn.mathutils.geometry import Sphere, sphere
from datasyn.optics.complens.tabular_lens import TabularLens, trace_tabular_lens
from datasyn.optics.imaging.pupil_function.sampling import (
    generate_random_input_rays,
    input_chief_ray,
    weighted_mean,
)
from datasyn.optics.paraxial import Field, Pupil
from datasyn.optics.ray import DEFAULT_WAVELENGTH, ORay
from datasyn.optics.typing import *


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class Rayfront:
    """
    TODO: Naming "rayfront" looks bad. Better name?
    """

    chief: ORay
    rays: ORay


class GenRayfrontOut(NamedTuple):
    rf: Rayfront
    chief_in: ORay
    rays_in: ORay
    p_xy: JArray  # Pupil coordinates


def generate_rayfront(
    lens: TabularLens,
    ep: Pupil,
    rng: RngKey,
    n: int,
    field: Field,
    f_xy: JArray,
    wvln: float = DEFAULT_WAVELENGTH,
    temp_disk_sampling: bool = True,
    temp_jitter: bool = False,
    verbose: bool = True,
    use_pss: bool = False,
):
    from datasyn.utils.context import maybe_ctx
    from datasyn.utils.time import easy_timer

    untraced = not debugjax.is_traced()
    verbose = untraced and verbose

    with maybe_ctx(easy_timer("generate input rays"), enabled=verbose):
        chief_in = input_chief_ray(ep, field, f_xy, wl=wvln)  # ()
        rays_in, p_xy = generate_random_input_rays(
            ep,
            rng,
            n,
            field,
            f_xy,
            wvln=wvln,
            jitter=temp_jitter,
            temp_disk_sampling=temp_disk_sampling,
            use_pss=use_pss,
        )  # (N,)

    def v_trace(ray_in: ORay):
        ray_out = trace_tabular_lens(lens, ray_in)
        return ray_out

    n_rays = rays_in.shape[0]

    with maybe_ctx(easy_timer(f"raytrace ({n_rays:,} rays)"), enabled=verbose):
        chief_out = v_trace(chief_in[None])  # (1,)
        chief_out = chief_out[0]  # ()
        rays_out = wrappings.chunk_vmapped(
            v_trace, chunk_size=2**20, disable_scan_trace=True
        )(rays_in)  # (N,)

    rf = Rayfront(chief=chief_out, rays=rays_out)
    return GenRayfrontOut(
        rf=rf,
        chief_in=chief_in,
        rays_in=rays_in,
        p_xy=p_xy,
    )


@jx.jit
def intersect_ray_sphere(cen: JArray, R: JArray, ray: ray.TRay):
    """
    Returns the ray propagation t of intersection with a sphere.

    NOTE: This allows both forward (t > 0) and backward (t < 0).
    """
    assert ray.ndim == 0

    d = ray.d
    p = ray.o - cen

    a = vecop.vecnormsqr(d)  # If the code is well-behaved, this must be always 1.
    b = 2 * vecop.vecdot(d, p)
    c = vecop.vecnormsqr(p) - R**2

    return safeop.quadratic_nonzero_a(a, b, c, order=False)


@jx.jit
def opd_to_phase_factor(wl: float, opd: FloatArray) -> ComplexArray:
    """
    More numerically robust: reduces OPD modulo wl before computing exp.

    TODO: I haven't checked when it's significantly more precise than non-robust version.
    """
    # phase = jnp.remainder(opd, wl) * ((2 * jnp.pi) / wl)  # in [0, 2π)
    phase = jnp.remainder(opd * ((2 * jnp.pi) / wl), 2 * jnp.pi)  # in [0, 2π)
    return jnp.exp(1j * phase)


@jax.tree_util.register_dataclass
@dataclass
class Wavefront:
    """
    TODO: Wavelength shape!
    """

    pts: JArray  # Sampled points, (*B, 3)
    mask: BoolArray  # Validity mask, (*B,)
    opd: JArray  # Optical path differences, (*B,)
    amp: JArray  # Amplitudes, (*B,)
    wvl: float  # Wavelength

    @property
    def shape(self):
        return self.opd.shape

    @property
    def ndim(self):
        return self.opd.ndim

    @property
    def dim(self):
        return self.pts.shape[-1]

    def __getitem__(self, idx):
        return Wavefront(
            pts=self.pts[idx], mask=self.mask[idx], opd=self.opd[idx], amp=self.amp[idx]
        )

    def reshape(self, shape):
        return Wavefront(
            pts=self.pts.reshape(*shape, self.dim),
            mask=self.mask.reshape(*shape),
            opd=self.opd.reshape(*shape),
            amp=self.amp.reshape(*shape),
            wvl=self.wvl,
        )

    def set_opd(self, opd: JArray):
        return Wavefront(
            pts=self.pts, mask=self.mask, opd=opd, amp=self.amp, wvl=self.wvl
        )

    def phase(self):
        """Calculates the phase. It's continuous and not normalized within [0, 2π)."""
        return (2 * jnp.pi) * (self.opd / self.wvl)

    def phase_factor(self):
        """Calculates the phase factor exp(ikr)."""
        return opd_to_phase_factor(self.wvl, self.opd)

    def wavefield(self):
        """Calculates the wave field A(r)exp(ikr)."""
        return self.amp * self.phase_factor()


class WavefrontByChiefOut(NamedTuple):
    wf: Wavefront
    rs: Sphere  # Reference sphere


@jx.jit
def _intersect_toward_reference_sphere(sph: Sphere, ray: ORay, xp_z: float):
    """
    Calculates the propagation from the image plane to the XP plane.

    TODO: Revise this logic!
    """
    assert ray.ndim == 0

    defined_1 = ray.valid

    # Criterion: Choose t that propagates to the closer point to the XP plane.

    # NOTE: (t1, t2) are either all defined or all undefined (because they are roots of a quadratic).
    (t1, t2), (defined_2, _) = intersect_ray_sphere(sph.c, sph.r, ray).vd
    z1 = jnp.abs(ray.o[..., 2] + (t1 * ray.d[..., 2:3]) - xp_z)
    z2 = jnp.abs(ray.o[..., 2] + (t2 * ray.d[..., 2:3]) - xp_z)
    idx = jnp.argmin(jnp.array([z1, z2]))
    t = jnp.array([t1, t2])[idx]

    defined = defined_1 & defined_2

    return t, defined


def make_xp_sphere(xp_z: float, pt_ref: JArray):
    """
    Reference sphere (RS): centered at a reference point and passing XP center
    """
    xp_cen = jnp.array([0.0, 0.0, xp_z])
    rs_R = safeop.norm(pt_ref - xp_cen).v
    return sphere(c=pt_ref, r=rs_R)


def wavefront_by_chief(rf: Rayfront, xp_z: float, img_z: float):
    """
    Constructs a wavefront on the reference sphere.
    The sphere is determined by the chief ray.
    """
    assert rf.chief.ndim == 0
    assert rf.rays.ndim == 1, "TODO: Arbitrary batch axes!"
    assert rf.rays.dim == rf.chief.dim

    n = 1.0  # TODO: Arbitrary medium!

    chief = rf.chief
    rays = rf.rays

    # Set the reference sphere (RS): centered at chief ray spot and passing XP center
    chief_spot = ray.project_to_z(img_z, chief).ray.o
    rs = make_xp_sphere(xp_z, chief_spot)

    # Propagate chief onto the RS.
    t_chief_rs, _ = _intersect_toward_reference_sphere(rs, chief, xp_z)
    opl_ref = chief.opl + (n * t_chief_rs)  # Chief's total OPL, ()

    # Propagate rays onto the RS.
    t_rays_rs, rays_def = jx.vmap(
        lambda r: _intersect_toward_reference_sphere(rs, r, xp_z)
    )(rays)  # (N,)
    opl_ray = rays.opl + (n * t_rays_rs)  # Ray's total OPL, (N,)

    # OPD is defined as the OPL difference between rays and chief.
    # NOTE: The sign is important.
    opd = opl_ray - opl_ref  # (N,)

    # NOTE: This might be meaningless!
    opd = jnp.where(rays_def, opd, 0.0)

    # Project points to the RS.
    pts = rays.propagate_o(t_rays_rs)

    # TODO: Revise amplitude!
    amp = jnp.ones_like(opd) * rays_def

    wf = Wavefront(pts=pts, mask=rays_def, opd=opd, amp=amp, wvl=chief.wvln)
    return WavefrontByChiefOut(wf=wf, rs=rs)


def wavefront_by_centroid(rf: Rayfront, xp_z: float, pt_foc: JArray):
    assert rf.chief.ndim == 0
    assert rf.rays.ndim == 1, "TODO: Arbitrary batch axes!"
    assert rf.rays.dim == rf.chief.dim

    n = 1.0  # TODO: Arbitrary medium!

    rays = rf.rays

    # Set the reference sphere (RS): centered at the given focal point and passing XP center
    rs = make_xp_sphere(xp_z, pt_foc)

    # Propagate rays onto the RS.
    t_rays_rs, rays_def = jx.vmap(
        lambda r: _intersect_toward_reference_sphere(rs, r, xp_z)
    )(rays)  # (N,)
    opl_ray = rays.opl + (n * t_rays_rs)  # Ray's total OPL, (N,)

    opl_mean = weighted_mean(opl_ray[:, None], rays_def).v[0]
    opd = opl_ray - opl_mean  # (N,)

    # NOTE: This might be meaningless!
    opd = jnp.where(rays_def, opd, 0.0)

    # Project points to the RS.
    pts = rays.propagate_o(t_rays_rs)

    # TODO: Revise amplitude!
    amp = jnp.ones_like(opd) * rays_def

    wf = Wavefront(pts=pts, mask=rays_def, opd=opd, amp=amp, wvl=rays[0].wvln)
    return WavefrontByChiefOut(wf=wf, rs=rs)
