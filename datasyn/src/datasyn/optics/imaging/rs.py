from typing import Optional, TypeAlias

import jax.numpy as jnp
import jaxtyping as jtp

import datasyn.optics.safeop as safeop
from datasyn.jaxutils import wrappings
from datasyn.optics.ray import DEFAULT_WAVELENGTH

FloatVec3Array1D: TypeAlias = jtp.Float[jtp.Array, "a 3"]
ComplexArray1D: TypeAlias = jtp.Complex[jtp.Array, "a"]


def rayleigh_sommerfeld(
    pts: FloatVec3Array1D,
    U0: ComplexArray1D,
    sensor_pos: FloatVec3Array1D,
    wvl: float = DEFAULT_WAVELENGTH,
    *,
    ns: Optional[FloatVec3Array1D] = None,
):
    """
    Rayleigh-Sommerfeld diffraction integral (Huygens-Fresnel principle)

    TODO: No proper scaling yet!

    P:     (N,3)
    wave:       (N,) wave
    sensor_pos: (M,3)
    """
    jk = 2j * jnp.pi / wvl
    N = pts.shape[0]

    assert pts.shape[:1] == U0.shape

    # TODO: Calculate this!
    dS = 1.0

    use_cosine = ns is not None

    def vf(u: FloatVec3Array1D):
        r = u[:, None] - pts[None]  # (B, N, 3)
        r_norm = safeop.norm(r).v  # (B, N)
        G = jnp.exp(jk * r_norm) / r_norm  # (B, N)

        if use_cosine:
            cos_theta = (r * ns[None]).sum(axis=2) / r_norm
            G = cos_theta * G

        U = (dS / (1j * wvl * N)) * jnp.sum(U0[None] * G, axis=1)

        return U

    U = wrappings.chunk_vmapped(vf, chunk_size=256)(sensor_pos)

    return U
