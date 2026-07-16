"""
Zernike polynomials with JAX.

This module provides Zernike polynomial evaluation using JAX for
GPU acceleration and automatic differentiation support.
"""

from dataclasses import dataclass
from enum import Enum
from functools import partial

import jax.numpy as jnp
from jax import Array

from datasyn.jaxutils import jx
from datasyn.optics.typing import *


@jax.tree_util.register_static
class ZernikeIndexing(Enum):
    """Indexing scheme for Zernike polynomials."""

    NOLL = "noll"
    # Future: ANSI, FRINGE, etc.


@jax.tree_util.register_static
class ZernikeKind(Enum):
    """Kind of Zernike polynomials (real or complex valued)."""

    REAL = "real"
    COMPLEX = "complex"


@jax.tree_util.register_static
class ZernikeNorm(Enum):
    """Normalization convention for Zernike polynomials."""

    NOLL = "noll"  # Unit variance over unit disk
    # Future: UNNORMALIZED, etc.


@jax.tree_util.register_static
@dataclass(frozen=True)
class ZernikeConfig:
    """
    TODO: Currently, this is not a valid JAX type!
          I think enum is problematic (not certain).

    Configuration for Zernike polynomial basis.

    This fully defines the polynomial basis including indexing,
    normalization, and whether real or complex valued.

    Attributes
    ----------
    n_max : int
        Maximum radial order.
    kind : ZernikeKind
        Real or complex valued polynomials.
    norm : ZernikeNorm
        Normalization convention.
    n_modes : int
        Total number of modes.
    ntab : Array
        Radial order n for each mode, shape (n_modes,).
    mtab : Array
        Azimuthal order m for each mode, shape (n_modes,).
    """

    n_max: int
    kind: ZernikeKind
    norm: ZernikeNorm
    n_modes: int
    ntab: tuple[int]
    mtab: tuple[int]

    @property
    def ntab_arr(self) -> IntArray:
        return jnp.asarray(self.ntab)

    @property
    def mtab_arr(self) -> IntArray:
        return jnp.asarray(self.mtab)


def _compute_noll_indices(n_max: int):
    """
    Compute (n, m) pairs in Noll ordering.

    References
    ----------
    R. Noll, "Zernike polynomials and atmospheric turbulence,"
    J. Opt. Soc. Am. 66, 207-211 (1976).
    """
    ntab = [0]
    mtab = [0]

    for ni in range(1, n_max + 1):
        mi = ni % 2
        while mi <= ni:
            if mi == 0:
                ntab.append(ni)
                mtab.append(0)
            else:
                j = len(ntab)
                if j % 2 == 1:
                    ntab.extend([ni, ni])
                    mtab.extend([mi, -mi])
                else:
                    ntab.extend([ni, ni])
                    mtab.extend([-mi, mi])
            mi += 2

    return tuple(ntab), tuple(mtab)


def _compute_norm_coefs(
    ntab: Sequence[int], mtab: Sequence[int], kind: ZernikeKind, norm: ZernikeNorm
) -> Array:
    """
    Compute normalization coefficients c_n^m.

    For Noll normalization:
        - Real: sqrt(2(n+1)) if m != 0, sqrt(n+1) if m == 0
        - Complex: sqrt(n+1)
    """
    n_modes = len(ntab)
    coefs = jnp.zeros(n_modes)

    for k, (n, m) in enumerate(zip(ntab, mtab)):
        if norm == ZernikeNorm.NOLL:
            if kind == ZernikeKind.COMPLEX:
                c = jnp.sqrt(n + 1.0)
            else:  # REAL
                c = jnp.sqrt(jnp.where(m == 0, 1, 2) * (n + 1.0))
        else:
            c = 1.0
        coefs = coefs.at[k].set(c)

    return coefs


def make_zernike_config(
    n_max: int,
    indexing: ZernikeIndexing = ZernikeIndexing.NOLL,
    kind: ZernikeKind = ZernikeKind.REAL,
    norm: ZernikeNorm = ZernikeNorm.NOLL,
) -> ZernikeConfig:
    """
    Create a Zernike polynomial configuration.

    Parameters
    ----------
    n_max : int
        Maximum radial order.
    indexing : ZernikeIndexing
        Indexing scheme. Currently only NOLL is supported.
    kind : ZernikeKind
        Real or complex valued polynomials.
    norm : ZernikeNorm
        Normalization convention.

    Returns
    -------
    ZernikeConfig
        Configuration fully defining the polynomial basis.
    """
    if indexing == ZernikeIndexing.NOLL:
        ntab, mtab = _compute_noll_indices(n_max)
    else:
        raise ValueError(f"Unsupported indexing: {indexing}")

    n_modes = len(ntab)

    return ZernikeConfig(
        n_max=n_max,
        kind=kind,
        norm=norm,
        n_modes=n_modes,
        ntab=ntab,
        mtab=mtab,
    )


def _eval_angular_real(mtab: IntArray, theta: Array) -> Array:
    """
    Evaluate angular part for real Zernike polynomials.

    Theta_n^m(theta) = cos(m*theta) if m >= 0, sin(|m|*theta) if m < 0
    """
    m_theta = theta[..., None] * mtab  # (..., n_modes)
    return jnp.where(mtab >= 0, jnp.cos(m_theta), jnp.sin(-m_theta))


def _eval_angular_complex(mtab: IntArray, theta: Array) -> Array:
    """
    Evaluate angular part for complex Zernike polynomials.

    Theta_n^m(theta) = exp(i*m*theta)
    """
    m_theta = theta[..., None] * mtab  # (..., n_modes)
    return jnp.exp(1j * m_theta)


@partial(jx.jit, static_argnames=("cfg",))
def eval_basis_zernipax(
    cfg: ZernikeConfig, rho: FloatArray, theta: FloatArray
) -> JArray:
    import zernipax as zpax

    ns, ms = cfg.ntab_arr, cfg.mtab_arr

    # rad = zpax.zernike_radial_cpu(rho, ns, ms)
    rad = zpax.zernike_radial_gpu(rho[:, None], ns[None], ms[None])

    match cfg.kind:
        case ZernikeKind.REAL:
            azi = _eval_angular_real(ms, theta)
        case ZernikeKind.COMPLEX:
            azi = _eval_angular_complex(ms, theta)

    norm = _compute_norm_coefs(ns, ms, cfg.kind, cfg.norm)

    polys = norm * rad * azi

    return polys
