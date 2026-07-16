from typing import NamedTuple

import jax.numpy as jnp

import datasyn.mathutils.geometry as geolib
import datasyn.optics.zernike as zernlib
from datasyn.jaxutils import debug as debugjax
from datasyn.jaxutils import jx
from datasyn.optics.complens.tabular_lens import TabularLens
from datasyn.optics.imaging.pupil_function.pf import (
    Wavefront,
    generate_rayfront,
    wavefront_by_centroid,
    wavefront_by_chief,
)
from datasyn.optics.paraxial import Field, Pupil
from datasyn.optics.ray import DEFAULT_WAVELENGTH
from datasyn.optics.typing import *
from datasyn.utils.time import easy_timer


@jx.jit
def cart2polar(xy: FloatArray):
    X, Y = xy[..., 0], xy[..., 1]

    r = jnp.sqrt(X * X + Y * Y)
    theta = jnp.arctan2(Y, X)
    theta = jnp.where(r == 0.0, 0.0, theta)

    return r, theta


@jx.jit
def polar2cart(r: FloatArray, t: FloatArray):
    x = r * jnp.cos(t)
    y = r * jnp.sin(t)
    return jnp.stack([x, y], axis=-1)


class ZernikeFitOut(NamedTuple):
    coefs: JArray
    polys: JArray
    coefs_pss: JArray
    polys_pss: JArray


def zernike_from_wavefront(
    wf: Wavefront,
    zer_cfg: zernlib.ZernikeConfig,
    xp_r: float,
    verbose: bool = True,
    zer_cfg_pss: zernlib.ZernikeConfig = None,
    pss: None | FloatArray = None,
):
    """
    TODO: Discuss about whether fit "phase" or "pupil function".
          I observed pupil function is not fitted well, but there were other issues of wavefront sampling.
          Currently, I'm not sure pupil function is still not fitted.
    """
    from datasyn.mathutils.linalg import masked_lstsq_lineax

    opd = wf.opd
    pts = wf.pts[..., 0:2]  # Projected xy coordinates
    mask = wf.mask

    pts_norm = pts / xp_r

    r, t = cart2polar(pts_norm)

    polys = zernlib.eval_basis_zernipax(zer_cfg, r, t)  # (#sample, #mode)

    coefs = masked_lstsq_lineax(polys, opd, mask)

    coefs_pss = None
    if pss is not None:
        # TODO: Quick-and-dirty to remove outliers (severely scattered)!
        mask_pss = pss < (1 * xp_r**2)
        polys_pss = zernlib.eval_basis_zernipax(zer_cfg_pss, r, t)  # (#sample, #mode)
        coefs_pss = masked_lstsq_lineax(polys_pss, pss, mask=mask_pss)

    if verbose:
        rec = jnp.sum(polys * coefs[None], axis=1)
        err = jnp.abs(rec - opd)[mask]
        print(f"Max error: {err.max():.3e}, Mean error: {err.mean()}")
        phase = wf.phase()
        print(f"Max phase difference: {(phase.max() - phase.min()):.1f}")

    return ZernikeFitOut(
        coefs=coefs, polys=polys, coefs_pss=coefs_pss, polys_pss=polys_pss
    )


class EasyZernikeOut(NamedTuple):
    coefs: JArray
    polys: JArray
    xp_sphere: geolib.Sphere
    wvl: float
    coefs_pss: None | FloatArray
    polys_pss: None | FloatArray


def easy_lens_to_zernike(
    lens: TabularLens,
    ep: Pupil,
    rng: RngKey,
    n: int,
    field: Field,
    f_xy: JArray,
    xp: Pupil,
    img_z: float,
    zer_cfg: zernlib.ZernikeConfig,
    zer_cfg_pss: zernlib.ZernikeConfig,
    wvl: float = DEFAULT_WAVELENGTH,
    pt_foc: None | JArray = None,
    verbose: bool = True,
    use_pss: bool = False,
):
    """
    Finds Zernike coefficients from scratch.
    """
    from datasyn.utils.context import maybe_ctx

    untraced = not debugjax.is_traced()
    verbose = untraced and verbose

    with maybe_ctx(easy_timer("Generate rayfront"), enabled=verbose):
        rf = generate_rayfront(
            lens,
            ep,
            rng,
            n,
            field,
            f_xy,
            wvln=wvl,
            verbose=verbose,
            use_pss=use_pss,
        ).rf

    with maybe_ctx(easy_timer("Rayfront -> Wavefront"), enabled=verbose):
        if pt_foc is None:
            out = wavefront_by_chief(rf, xp.z, img_z=img_z)
        else:
            out = wavefront_by_centroid(rf, xp.z, pt_foc=pt_foc)
        wf = out.wf
        xp_sphere = out.rs

    with maybe_ctx(easy_timer("Wavefront -> Zernike"), enabled=verbose):
        out = zernike_from_wavefront(
            wf,
            zer_cfg,
            xp.r,
            verbose=verbose,
            zer_cfg_pss=zer_cfg_pss,
            pss=rf.rays._pss,
        )

    return EasyZernikeOut(
        coefs=out.coefs,
        polys=out.polys,
        xp_sphere=xp_sphere,
        wvl=wvl,
        coefs_pss=out.coefs_pss,
        polys_pss=out.polys_pss,
    )
