from dataclasses import dataclass
from functools import partial

import jax.numpy as jnp

import datasyn.jaxutils.nputils as myjnputils
import datasyn.mathutils.geometry as geolib
import datasyn.optics.zernike as zernlib
from datasyn.jaxutils import jx
from datasyn.optics.imaging.defocus import SphereWavefront
from datasyn.optics.imaging.pupil_function.pf import opd_to_phase_factor
from datasyn.optics.typing import *
from datasyn.utils.time import easy_timer


@dataclass(frozen=True)
class ZernikeDebyeWavefront(SphereWavefront):
    sp: geolib.Sphere
    cfg: zernlib.ZernikeConfig
    coefs: JArray
    wvl: float
    na: float

    @property
    def sphere(self):
        return self.sp

    def sample(self, theta: FloatArray, phi: FloatArray):
        import datasyn.optics.zernike as zern

        # TODO: Avoid hard-coded reshaping!
        rho = jnp.sin(theta) / self.na
        rho_flat = rho.reshape(-1)
        phi_flat = phi.reshape(-1)
        basis = zern.eval_basis_zernipax(self.cfg, rho_flat, phi_flat)
        basis = myjnputils.unflatten(basis, 0, rho.shape)

        # TODO: Avoid hard-coded broadcasting!
        zer = jnp.sum(self.coefs[None, None] * basis, axis=-1)

        amp = 1.0
        pf = opd_to_phase_factor(self.wvl, zer)
        wave = amp * pf

        return wave


def fn_sample_zernike_wavefront(
    wvl: float,
    rho: FloatArray,
    phi: FloatArray,
    coefs: JArray,
    cfg: zernlib.ZernikeConfig,
    coefs_pss: None | JArray = None,
    cfg_pss: zernlib.ZernikeConfig = None,
    disable_timer: bool = True,
):
    import datasyn.optics.zernike as zern

    def jax_barrier():
        if not disable_timer:  # To measure time
            jx.block_until_ready_all()

    # TODO: Avoid hard-coded reshaping!
    jax_barrier()
    with easy_timer("eval_basis_zernipax", disable=disable_timer):
        rho_flat = rho.reshape(-1)
        phi_flat = phi.reshape(-1)
        basis = zern.eval_basis_zernipax(cfg, rho_flat, phi_flat)
        basis = myjnputils.unflatten(basis, 0, rho.shape)
        jax_barrier()

    jax_barrier()
    with easy_timer("wavefront computation", disable=disable_timer):
        # TODO: Avoid hard-coded broadcasting!
        zer = jnp.sum(coefs[None, None] * basis, axis=-1)

        amp = 1.0
        pf = opd_to_phase_factor(wvl, zer)
        wave = amp * pf
        jax_barrier()

    if coefs_pss is not None:
        jax_barrier()
        with easy_timer("pss computation", disable=disable_timer):
            basis_pss = zern.eval_basis_zernipax(cfg_pss, rho_flat, phi_flat)
            basis_pss = myjnputils.unflatten(basis_pss, 0, rho.shape)
            # TODO: Avoid hard-coded broadcasting!
            pss = jnp.sum(coefs_pss[None, None] * basis_pss, axis=-1)
            wave = (pss <= 0.0) * wave
            jax_barrier()

    return wave
