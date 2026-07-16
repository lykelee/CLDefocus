from dataclasses import dataclass
from functools import partial
from typing import Any, Generic, NamedTuple, Tuple, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jr

import datasyn.mathutils.geometry as geolib
import datasyn.mathutils.grid as gridlib
import datasyn.optics.paraxial as prx
import datasyn.optics.zernike as zernlib
from datasyn.jaxutils.typing import JArray
from datasyn.optics.complens.tabular_lens import TabularLens
from datasyn.optics.imaging.pupilmodel import PupilModel


def wh_physics_to_image(x, axis_w: int, axis_h: int):
    # Physics: (w, h) h-axis up. Image: (h, w) h-axis flipped.
    return jnp.swapaxes(jnp.flip(x, axis_h), axis_w, axis_h)


import datasyn.optics.ray as rayopt
import datasyn.optics.safeop as safeop
from datasyn.jaxutils import jx
from datasyn.optics.imaging.pupil_function.zernike_fit import EasyZernikeOut
from datasyn.optics.ray import DEFAULT_WAVELENGTH
from datasyn.utils.context import maybe_ctx
from datasyn.utils.time import easy_timer


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class ImagingHelper:
    pm: PupilModel
    tanhfov: float
    objfwd: int  # 1 or -1

    def dist_to_field(self, d: JArray):
        field_z = self.pm.ep.z + self.objfwd * d
        field_r = d * self.tanhfov
        return prx.Field(z=field_z, r=field_r)

    @property
    def hfov(self):
        return jnp.arctan(self.tanhfov)

    @property
    def fov(self):
        return 2 * jnp.arctan(self.tanhfov)

    def tan_of_field(self, f_xy: float | Tuple[float, float]):
        f = safeop.norm(jnp.asarray(f_xy)).v
        return f * self.tanhfov

    def angle_of_field(self, f_xy: float | Tuple[float, float]):
        return jnp.arctan(self.tan_of_field(f_xy))

    def field_of_tan(self, tan: float):
        return tan / self.tanhfov

    def field_of_angle(self, angle: float):
        return self.field_of_tan(jnp.tan(angle))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Parax:
    rtm: prx.RTM2
    ep: prx.Pupil
    xp: prx.Pupil

    @property
    def pupmag(self):
        return self.xp.r / self.ep.r

    @property
    def efl(self):
        return self.rtm.efl()

    @property
    def fnum(self):
        return self.efl / (2 * self.ep.r)

    def obj2img_rel(self, s_obj: float):
        """Relative to EP, XP"""
        m = self.pupmag
        return m**2 / (m * self.rtm.power() - 1 / s_obj)

    def img2obj_rel(self, s_img: float):
        """Relative to EP, XP"""
        m = self.pupmag
        return 1 / (m * (self.rtm.power() - m / s_img))


class TiltedDebyeCoords(NamedTuple):
    nasin: float
    foc: float
    tilt_theta: float
    tilt_phi: float
    s_img: float
    s_obs: float

    @property
    def z_defocus(self):
        return self.s_obs - self.s_img

    @property
    def w_defocus(self):
        return (self.s_obs - self.s_img) / jnp.cos(self.tilt_theta)

    @property
    def na_angle(self):
        return jnp.arcsin(self.nasin)


def make_tilted_debye_coords(
    xp: prx.Pupil,
    xp_sphere: geolib.Sphere,
    z_obs: float,
):
    # Tilt
    vert = safeop.norm(xp_sphere.c[0:2]).v
    tilt_theta = jnp.arctan(jnp.abs(vert / (xp_sphere.c[2] - xp.z)))
    tilt_phi = 0.0  # TODO: Probably wrong!

    # NA cap
    f = xp_sphere.r
    sin = safeop.sqrt(xp.r**2 / (xp.r**2 + f**2)).v

    # Defocus
    s_img = xp_sphere.c[2] - xp.z
    s_obs = z_obs - xp.z

    return TiltedDebyeCoords(
        nasin=sin,
        foc=f,
        tilt_theta=tilt_theta,
        tilt_phi=tilt_phi,
        s_img=s_img,
        s_obs=s_obs,
    )


def debye_sampling_bound_for_antialiasing_from_coords(
    ior: float, wvl: float, coords: TiltedDebyeCoords
):
    from datasyn.optics.imaging.debye import debye_sampling_bound_for_antialiasing

    return debye_sampling_bound_for_antialiasing(
        ior=ior, wvl=wvl, sin=coords.nasin, z=coords.w_defocus
    )


def _zernike_tilted_debye(
    wvl: float,
    xp: prx.Pupil,
    xp_sphere: geolib.Sphere,
    z_obs: float,
    sen: gridlib.BoundedGrid,
    ks: int,
    zer_cfg: zernlib.ZernikeConfig,
    zer_coefs: JArray,
    MM: int = 64,
    verbose: int = 2,
    zer_coefs_pss: None | JArray = None,
    zer_cfg_pss: zernlib.ZernikeConfig = None,
):
    from datasyn.optics.imaging.debye import tilted_debye

    ior = 1.0  # IOR of image space
    upsample = 5

    # Coordinates (TODO: Separate this to another function!)

    # Tilt
    vert = safeop.norm(xp_sphere.c[0:2]).v
    theta_0 = jnp.arctan(jnp.abs(vert / (xp_sphere.c[2] - xp.z)))
    invcost = 1 / jnp.cos(theta_0)

    phi_0 = 0.0

    # NA cap
    f = xp_sphere.r  # focal length
    sin = safeop.sqrt(xp.r**2 / (xp.r**2 + f**2)).v

    # Defocus
    w = invcost * (z_obs - xp_sphere.c[2])

    vport_xy_world = w * jnp.array([jnp.tan(theta_0), 0.0])
    vport_xy_pix = sen.quantize(vport_xy_world).index
    hks = ks // 2
    vport = sen.slice((vport_xy_pix[0] - hks, vport_xy_pix[1] - hks), (ks, ks))

    # Tilted Debye CZT

    I_out = tilted_debye(
        wvl=wvl,
        ior=ior,
        sin=sin,
        theta_0=theta_0,
        phi_0=phi_0,
        w=w,
        upsample=upsample,
        vport=vport,
        MM=MM,
        zer_cfg=zer_cfg,
        zer_coefs=zer_coefs,
        zer_cfg_pss=zer_cfg_pss,
        zer_coefs_pss=zer_coefs_pss,
        verbose=verbose,
    )

    return I_out, vport, vport_xy_world


TWavefront = TypeVar("TWavefront")


class PropOut(NamedTuple):
    I_psf: JArray
    vport: gridlib.BoundedGrid
    vport_xy_world: JArray
    scoc: float
    failed: bool = False


class Proper(Generic[TWavefront]):
    def generate_wavefront(
        self,
        lens: TabularLens,
        parax: Parax,
        field: prx.Field,
        f_xy: JArray,
        wvl: float,
    ) -> TWavefront:
        """
        Generate wavefront or something similar to this concept.
        It depends on contexts. For some method it would be direct ray samples,
        while it might be Zernike polynomials for some other methods.
        """
        ...

    def propagate_wavefront(
        self,
        wf: TWavefront,
        s_prop: float,
        parax: Parax,
        sen: gridlib.BoundedGrid,
        ks: int,
        **kwargs,
    ) -> PropOut:
        """
        Propagate the wavefront to render PSF.
        """
        ...


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DebyeProper(Proper[EasyZernikeOut]):
    zer_cfg: zernlib.ZernikeConfig
    zer_cfg_pss: zernlib.ZernikeConfig = None

    def __post_init__(self):
        if self.zer_cfg_pss is None:
            zer_n_max = 5
            zer_cfg = zernlib.make_zernike_config(
                n_max=zer_n_max, kind=zernlib.ZernikeKind.REAL
            )
            object.__setattr__(self, "zer_cfg_pss", zer_cfg)

    def generate_wavefront(
        self,
        lens: TabularLens,
        parax: Parax,
        field: prx.Field,
        f_xy: JArray,
        wvl: float,
    ):
        from datasyn.optics.imaging.pupil_function.sampling import (
            lens_best_focus_by_intersection,
        )
        from datasyn.optics.imaging.pupil_function.zernike_fit import (
            easy_lens_to_zernike,
        )

        z_start = lens.get_z(0) - 1e0

        pt_foc = lens_best_focus_by_intersection(
            lens=lens,
            z_start=z_start,
            ep=parax.ep,
            rng=jr.PRNGKey(0),
            field=field,
            f_xy=f_xy,
            wvl=wvl,
        )
        zern_out = easy_lens_to_zernike(
            lens=lens,
            ep=parax.ep.scale(1.2),
            rng=jr.PRNGKey(0),
            n=64,
            field=field,
            f_xy=f_xy,
            xp=parax.xp,
            img_z=pt_foc[2],
            zer_cfg=self.zer_cfg,
            zer_cfg_pss=self.zer_cfg_pss,
            wvl=wvl,
            pt_foc=pt_foc,
            verbose=False,
            use_pss=True,
        )

        return zern_out

    def propagate_wavefront(
        self,
        wf: EasyZernikeOut,
        s_prop: float,
        parax: Parax,
        sen: gridlib.BoundedGrid,
        ks: int,
        MM: int = 1024,
    ):
        disable_timer = True

        def jax_barrier():
            if not disable_timer:  # To measure time
                jx.block_until_ready_all()

        z_prop = parax.xp.z + s_prop

        jax_barrier()
        with maybe_ctx(
            easy_timer(f"propagate_wavefront - _zernike_tilted_debye"),
            enabled=not disable_timer,
        ):
            I_psf, vport, vport_xy_world = _zernike_tilted_debye(
                wvl=wf.wvl,
                xp=parax.xp,
                xp_sphere=wf.xp_sphere,
                z_obs=z_prop,
                sen=sen,
                ks=ks,
                zer_cfg=self.zer_cfg,
                zer_coefs=wf.coefs,
                MM=MM,
                verbose=2 if not disable_timer else 0,
                zer_coefs_pss=wf.coefs_pss,
                zer_cfg_pss=self.zer_cfg_pss,
            )
            jax_barrier()

        dcoords = make_tilted_debye_coords(parax.xp, wf.xp_sphere, z_obs=z_prop)
        scoc = rough_scoc_debye_from_coords(dcoords)

        # Research NOTE 260418:
        # I found computed PSFs sometimes have large nonzero regions.
        # Although they don't almost appear in visualization, they cause artifact for imaging.
        # I don't know why they appear, so I adopted this quick-and-dirty.
        jax_barrier()
        with maybe_ctx(
            easy_timer(f"propagate_wavefront - Gaussian masking"),
            enabled=not disable_timer,
        ):
            from datasyn.optics.imaging.analysis import gaussian_kernel

            pixsize = sen.grid.spacing
            w = gaussian_kernel(
                ks=ks,
                mean=jnp.zeros((2,)),
                cov=jnp.diag((1.5 * scoc / pixsize) ** 2),
                normalize=False,  # To reduce redundant computation
            )
            I_psf = w * I_psf

            # Research NOTE 260422:
            # Occasionally PSF computations fail, yielding empty array.
            # To avoid this, we provide an identity fallback.
            out = safeop.div(I_psf, I_psf.sum())
            I_psf = out.v
            failed = ~out.d
            iden = jnp.zeros_like(I_psf).at[ks // 2, ks // 2].set(1.0)
            I_psf = jnp.where(failed, iden, out.v)

            jax_barrier()

        return PropOut(
            I_psf=I_psf,
            vport=vport,
            vport_xy_world=vport_xy_world,
            scoc=scoc,
            failed=failed,
        )


def rough_scoc_debye(na_angle: float, chief_angle: float, z_defocus: float):
    return z_defocus * jnp.abs(
        jnp.sin(na_angle)
        / (
            jnp.cos(chief_angle + 0.5 * na_angle)
            * jnp.cos(chief_angle - 0.5 * na_angle)
        )
    )


def rough_scoc_debye_from_coords(coords: TiltedDebyeCoords):
    return rough_scoc_debye(
        na_angle=coords.na_angle,
        chief_angle=coords.tilt_theta,
        z_defocus=coords.z_defocus,
    )


from datasyn.optics.imaging.pupil_function.pf import Wavefront


class HFWavefront(NamedTuple):
    """
    NOTE: Codes relevant this topic are now highly messy!
          We need to reorganize all of them in someday.
    """

    wf: Wavefront
    xp_sphere: geolib.Sphere  # Reference sphere
    chief: rayopt.ORay


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class HFProper(Proper[HFWavefront]):
    """
    Huygens-Fresnel for spherical surface
    """

    def generate_wavefront(
        self,
        lens: TabularLens,
        parax: Parax,
        field: prx.Field,
        f_xy: JArray,
        wvl: float,
    ):
        from datasyn.optics.imaging.pupil_function.pf import (
            generate_rayfront,
            wavefront_by_centroid,
        )
        from datasyn.optics.imaging.pupil_function.sampling import (
            lens_best_focus_by_intersection,
        )

        n = 1024
        rng = jr.PRNGKey(0)
        z_start = lens.get_z(0) - 1e0

        pt_foc = lens_best_focus_by_intersection(
            lens=lens,
            z_start=z_start,
            ep=parax.ep,
            rng=jr.PRNGKey(0),
            field=field,
            f_xy=f_xy,
            wvl=wvl,
        )
        rf = generate_rayfront(
            lens, parax.ep, rng, n, field, f_xy, wvln=wvl, verbose=False
        ).rf
        out = wavefront_by_centroid(rf, parax.xp.z, pt_foc=pt_foc)

        return HFWavefront(wf=out.wf, xp_sphere=out.rs, chief=rf.chief)

    def propagate_wavefront(
        self,
        wf: HFWavefront,
        s_prop: float,
        parax: Parax,
        sen: gridlib.BoundedGrid,
        ks: int,
    ):
        import datasyn.jaxutils.nputils as myjnputils
        import datasyn.mathutils.vecop as vecop
        from datasyn.optics.imaging.defocus import downsample_integer
        from datasyn.optics.imaging.pupil_function.pf import (
            opd_to_phase_factor,
            sqabs_complex,
        )
        from datasyn.optics.imaging.rs import rayleigh_sommerfeld
        from datasyn.optics.ray import project_to_z

        z_prop = parax.xp.z + s_prop
        vport_xy_world = project_to_z(z_prop, wf.chief).ray.o[0:2]
        vport_xy_pix = sen.quantize(vport_xy_world).index
        hks = ks // 2
        vport = sen.slice((vport_xy_pix[0] - hks, vport_xy_pix[1] - hks), (ks, ks))

        upsample = 3
        vport_fine = vport.upsample(upsample)

        eval_pts = myjnputils.flatten(vport_fine.grid_points(), 0, 2)
        eval_pts = vecop.xy2xyz(eval_pts, z_prop)

        wvl = wf.wf.wvl
        amp = 1.0 * wf.wf.amp
        pf = opd_to_phase_factor(wvl, wf.wf.opd)
        wave = amp * pf

        xp_sphere = wf.xp_sphere
        ns = safeop.normdir(xp_sphere.c[None] - wf.wf.pts).v

        U_out_fine = rayleigh_sommerfeld(wf.wf.pts, wave, eval_pts, wvl, ns=ns)

        U_out_fine = myjnputils.unflatten(U_out_fine, 0, vport_fine.shape)
        I_out_fine = sqabs_complex(U_out_fine)
        I_out = downsample_integer(I_out_fine, upsample)

        return PropOut(
            I_psf=I_out, vport=vport, vport_xy_world=vport_xy_world, scoc=None
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Imaging:
    lens: TabularLens
    sen: gridlib.BoundedGrid
    parax: Parax
    imghelp: ImagingHelper
    wvl_ref: float
    proper: Proper

    @property
    def z_enter(self):
        return self.parax.rtm.z_enter

    @property
    def ep(self):
        return self.parax.ep

    @property
    def xp(self):
        return self.parax.xp

    def calc_sensor_s(self, focusing: float, f_xy: JArray, wvl: float | None = None):
        from datasyn.optics.imaging.pupil_function.sampling import (
            lens_best_focus_by_intersection,
        )

        if wvl is None:
            wvl = self.wvl_ref

        z_start = self.z_enter - 1e0
        field_foc = self.imghelp.dist_to_field(focusing)

        z_prop = lens_best_focus_by_intersection(
            lens=self.lens,
            z_start=z_start,
            ep=self.ep,
            rng=jr.PRNGKey(0),
            field=field_foc,
            f_xy=f_xy,
            wvl=wvl,
        )[2]

        s_prop = z_prop - self.xp.z
        return s_prop

    def generate_wavefront(
        self,
        depth: float,
        f_xy: JArray,
        wvl: float | None = None,
    ):
        if wvl is None:
            wvl = self.wvl_ref

        field = self.imghelp.dist_to_field(depth)

        return self.proper.generate_wavefront(
            lens=self.lens, parax=self.parax, field=field, f_xy=f_xy, wvl=wvl
        )

    def propagate_wavefront(
        self,
        wf_out: Any,
        s_prop: float,
        ks: int,
        **kwargs,
    ):
        return self.proper.propagate_wavefront(
            wf_out, s_prop, parax=self.parax, sen=self.sen, ks=ks, **kwargs
        )


def bool2sign(is_positive: jnp.ndarray, dtype=jnp.int32) -> JArray:
    """
    True -> +1, False -> -1
    """
    return 2 * jnp.astype(is_positive, dtype) - dtype(1)


def make_imaging_helper_260403(parax: Parax, tanhfov: float, backward: bool = False):
    pm = PupilModel(ep=parax.ep, xp=parax.xp, op=parax.rtm.power())
    objfwd = bool2sign(backward)
    return ImagingHelper(pm=pm, tanhfov=tanhfov, objfwd=objfwd)


def _make_imaging_helper(
    lens: TabularLens,
    wvl_ref: float = DEFAULT_WAVELENGTH,
):
    rtm, ep, xp = lens.calc_paraxial_properties(wvl_ref)
    parax = Parax(rtm=rtm, ep=ep, xp=xp)

    tanhfov_imgside = lens.imgrad / jnp.abs(rtm.bfp() - xp.z)
    tanhfov = (xp.r / ep.r) * tanhfov_imgside

    imghelp = make_imaging_helper_260403(parax, tanhfov)

    return parax, imghelp


def make_imaging(
    lens: TabularLens,
    sen_wh: Tuple[int, int],
    proper: Proper,
    wvl_ref: float = DEFAULT_WAVELENGTH,
    pixsize: float | None = None,
):
    parax, imghelp = _make_imaging_helper(lens, wvl_ref=wvl_ref)

    Nx, Ny = sen_wh

    if pixsize is None:
        from datasyn.optics.imaging.defocus import make_sensor_from_imgrad

        sen = make_sensor_from_imgrad(Nx=Nx, Ny=Ny, imgrad=lens.imgrad).to_grid()

    else:
        from datasyn.optics.imaging.defocus import make_sensor_from_pixsize

        sen = make_sensor_from_pixsize(Nx=Nx, Ny=Ny, pixsize=pixsize).to_grid()

    return Imaging(
        lens=lens,
        sen=sen,
        parax=parax,
        imghelp=imghelp,
        wvl_ref=wvl_ref,
        proper=proper,
    )


def make_imaging_helper(
    lens: TabularLens,
    wvl_ref: float = DEFAULT_WAVELENGTH,
):
    _, imghelp = _make_imaging_helper(lens, wvl_ref=wvl_ref)
    return imghelp
