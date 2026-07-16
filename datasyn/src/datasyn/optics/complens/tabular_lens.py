"""
Implementation of compound lenses optimized for JAX.

This module provides a highly optimized lens representation for "tabular form" lenses
using even symmetric polynomial aspheric surfaces. All parameters are stored in
JAX arrays for efficient tracing with `jax.lax.fori_loop`.

The structure mirrors the mytable format:
- SURFACES table -> surfaces array (c, z, aperture, CauchyA, CauchyB)
- ASPHERIC table -> asph_coeffs array (K, A4, A6, ...)
- STOP -> stop_idx integer
- SHAPE -> asph_map array (-1 for spherical, >=0 for aspheric row index)
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

import datasyn.mathutils.vecop as vecop
import datasyn.optics.paraxial as parax
import datasyn.optics.ray as ray
import datasyn.optics.ray as rayopt
from datasyn.jaxutils import jx
from datasyn.optics.complens.surfaces import (
    intersect_even_asphere,
    intersect_spherical,
    refract,
)
from datasyn.optics.typing import *
from datasyn.optics.typing import JArray

# Standard wavelengths (Schott standard) for the d/F/C lines, in micrometers.
_WAVE_D_UM = 0.5875618
_WAVE_F_UM = 0.4861327
_WAVE_C_UM = 0.6562725


@jx.jit
def get_cauchy_2term(nd, vd):
    """
    Convert index (Nd) and Abbe number (Vd) to 2-term Cauchy coefficients (A, B).

    n(lambda) = A + B / lambda^2
    """
    dispersion = (nd - 1.0) / vd
    k_factor = (1.0 / _WAVE_F_UM**2) - (1.0 / _WAVE_C_UM**2)
    b_coeff = dispersion / k_factor
    a_coeff = nd - (b_coeff / _WAVE_D_UM**2)
    return a_coeff, b_coeff


# Column indices for surfaces array
class SurfCol:
    """Column indices for the surfaces array."""

    C = 0  # Curvature (1/R)
    Z = 1  # Cumulative z-position
    APER = 2  # Aperture radius
    CAUCHY_A = 3  # Cauchy coefficient A
    CAUCHY_B = 4  # Cauchy coefficient B
    N_COLS = 5


class TabularLens(NamedTuple):
    """
    Tabular-form compound lens with even polynomial aspheric surfaces.

    This is optimized for JAX-based ray tracing using `jax.lax.fori_loop`.
    All parameters are stored in arrays mirroring the `.mytable` structure.

    Attributes:
        surfaces: (n_surfaces, 5) float array
            Columns: [c, z, aperture, CauchyA, CauchyB]
            - c: curvature (1/R), 0 for flat
            - z: cumulative z-position from first surface
            - aperture: aperture semi-diameter
            - CauchyA, CauchyB: Cauchy dispersion coefficients for material
              AFTER this surface. Material BEFORE surface 0 is Air.

        asph_map: (n_surfaces,) int32 array
            Maps surface index to aspheric coefficient row.
            -1 = spherical surface (no aspheric terms)
            >=0 = row index in asph_coeffs array

        asph_coeffs: (n_asph, 1 + n_terms) float array
            Aspheric coefficients for aspheric surfaces only.
            Columns: [K, A4, A6, A8, ...] where K is conic constant.
            Number of terms (A4, A6, ...) is lens-dependent.
            Empty array (shape (0, 1)) if no aspheric surfaces.

        stop_idx: Index of the aperture stop surface.

    Example:
        For a lens with 10 surfaces where surfaces 2 and 5 are aspheric:
        - surfaces.shape = (10, 5)
        - asph_map = [-1, -1, 0, -1, -1, 1, -1, -1, -1, -1]
        - asph_coeffs.shape = (2, 1 + n_terms)  # rows for surfaces 2 and 5

    Notes:
        - Even polynomial aspheric sag: z(r) = c*r^2/(1 + sqrt(1-(1+K)*c^2*r^2))
          + A4*r^4 + A6*r^6 + A8*r^8 + ...
        - Material IOR at wavelength λ: n(λ) = A + B/λ^2 (Cauchy 2-term)
        - All units should be consistent (typically meters).
    """

    surfaces: JArray  # (n_surfaces, 5)
    asph_map: JArray  # (n_surfaces,) int32
    asph_coeffs: JArray  # (n_asph, 1 + n_terms)
    stop_idx: int
    imgrad: float
    imgpos: float  # NOTE: This is now for test!

    @property
    def n_surfaces(self) -> int:
        """Number of surfaces in the lens."""
        return self.surfaces.shape[0]

    @property
    def n_asph(self) -> int:
        """Number of aspheric surfaces."""
        return self.asph_coeffs.shape[0]

    @property
    def max_asph_order(self) -> int:
        """Maximum aspheric polynomial order (4, 6, 8, ...)."""
        # asph_coeffs has columns [K, A4, A6, ...], so n_terms = n_cols - 1
        # max_order = 4 + 2*(n_terms - 1) = 2 + 2*n_terms = 2*n_cols
        n_cols = self.asph_coeffs.shape[1]
        if n_cols <= 1:
            return 0
        return 2 + 2 * (n_cols - 1)

    def get_curvature(self, i: int) -> JArray:
        """Get curvature of surface i."""
        return self.surfaces[i, SurfCol.C]

    def get_z(self, i: int) -> JArray:
        """Get z-position of surface i."""
        return self.surfaces[i, SurfCol.Z]

    def get_aperture(self, i: int) -> JArray:
        """Get aperture radius of surface i."""
        return self.surfaces[i, SurfCol.APER]

    def get_cauchy(self, i: int) -> tuple[JArray, JArray]:
        """Get Cauchy coefficients (A, B) for material AFTER surface i."""
        return self.surfaces[i, SurfCol.CAUCHY_A], self.surfaces[i, SurfCol.CAUCHY_B]

    def get_ior(self, i: int, wvln: JArray) -> JArray:
        A, B = self.get_cauchy(i)
        return _cauchy2_ior(A, B, wvln)

    def is_aspheric(self, i: int) -> JArray:
        """Check if surface i is aspheric."""
        return self.asph_map[i] >= 0

    def get_asph_coeffs(self, i: int) -> JArray:
        """
        Get aspheric coefficients [K, A4, A6, ...] for surface i.

        Returns zeros if surface is spherical.
        """
        asph_idx = self.asph_map[i]
        # If spherical (asph_idx < 0), return zeros
        n_cols = self.asph_coeffs.shape[1]
        return jax.lax.cond(
            asph_idx >= 0,
            lambda: self.asph_coeffs[asph_idx],
            lambda: jnp.zeros(n_cols),
        )

    def get_conic(self, i: int) -> JArray:
        """Get conic constant K for surface i (0 if spherical)."""
        coeffs = self.get_asph_coeffs(i)
        return coeffs[0]

    def get_poly_coeffs(self, i: int) -> JArray:
        """Get polynomial coefficients [A4, A6, ...] for surface i."""
        coeffs = self.get_asph_coeffs(i)
        return coeffs[1:]

    def get_paraxial_sequence(self, wvln: float):
        n_surf = self.n_surfaces

        cs = self.surfaces[1:, SurfCol.C]
        ns = jx.vmap(lambda i: self.get_ior(i, wvln))(jnp.arange(0, n_surf))
        zs = self.surfaces[0:, SurfCol.Z]
        ts = jnp.diff(zs[1:])  # Convert z to thickness
        z0 = zs[1]

        return ns, cs, ts, z0

    def calc_rtm(self, wvln: float):
        ns, cs, ts, z0 = self.get_paraxial_sequence(wvln)
        return parax.rtm2.ior_curv_thick_seq(ns, cs, ts, z0)

    def calc_paraxial_properties(self, wvln: float):
        """
        NOTE: This is quite confusing part!
        """
        from datasyn.optics.complens.paraxial import find_ep_xp

        ns, cs, ts, z0 = self.get_paraxial_sequence(wvln)
        st = self.stop_idx

        rtm_front = parax.rtm2.ior_curv_thick_seq(
            ns[: st + 1], cs[:st].at[-1].set(0.0), ts[: st - 1], z0
        )
        rtm_stop = parax.rtm2.ior_curv_thick_seq(
            ns[st - 1 : st + 1], cs[st - 1 : st], ts[:0], rtm_front.z_exit
        )
        rtm_back = parax.rtm2.ior_curv_thick_seq(
            ns[st - 1 :], cs[st - 1 :].at[0].set(0.0), ts[st - 1 :], rtm_front.z_exit
        )
        stop_radius = self.get_aperture(st)
        n_obj = ns[0]
        n_img = ns[-1]

        M = rtm_back.M @ rtm_stop.M @ rtm_front.M
        rtm = parax.RTM2(M=M, z_enter=rtm_front.z_enter, z_exit=rtm_back.z_exit)
        ep, xp = find_ep_xp(rtm_front, rtm_back, stop_radius, n_obj, n_img)

        return rtm, ep, xp


@jx.jit
def _cauchy2_ior(A: JArray, B: JArray, wvln: JArray) -> JArray:
    """
    TODO: Replace with my existing one!

    Compute IOR using 2-term Cauchy formula: n = A + B/λ²

    Args:
        A: Cauchy A coefficient
        B: Cauchy B coefficient (in µm² units)
        wvln: Wavelength in meters
    """
    # Convert wavelength from meters to micrometers
    wvln_um = wvln * 1e6
    return A + B / (wvln_um**2)


def trace_tabular_lens(
    lens: TabularLens,
    input_ray: rayopt.TLensRay,
) -> rayopt.TLensRay:
    """
    Trace rays through a tabular-form lens using `lax.fori_loop`.

    Args:
        lens: TabularLens instance
        input_ray: Input TLensRay (must have wavelength field)

    Returns:
        Output TLensRay after tracing through all surfaces
    """
    n_surfaces = lens.n_surfaces
    has_asph = lens.n_asph > 0
    wvln = input_ray.wvln

    def trace_single_surface(si: int, ray_cur: rayopt.TLensRay) -> rayopt.TLensRay:
        surf_row = lens.surfaces[si]
        c = surf_row[SurfCol.C]
        z_surf = surf_row[SurfCol.Z]
        aperture = surf_row[SurfCol.APER]

        # Calculate IORs between two media.
        # TODO: Pre-compute outside loop? (though it may not be significant)
        cauchy_A_before = lens.surfaces[si - 1, SurfCol.CAUCHY_A]
        cauchy_B_before = lens.surfaces[si - 1, SurfCol.CAUCHY_B]
        cauchy_A_after = lens.surfaces[si, SurfCol.CAUCHY_A]
        cauchy_B_after = lens.surfaces[si, SurfCol.CAUCHY_B]
        n_before = _cauchy2_ior(cauchy_A_before, cauchy_B_before, wvln)
        n_after = _cauchy2_ior(cauchy_A_after, cauchy_B_after, wvln)

        # Begin to trace rays on the current surface.

        # 1. Translate ray to local coordinates (surface at z=0).
        ray_local = ray.translate(ray_cur, jnp.array([0.0, 0.0, -z_surf]))

        # 2. Find intersection with surface.

        def trace_sph():
            out = intersect_spherical(c, ray_local, n_before)
            return out.ray, out.normal, out.defined

        if has_asph:
            asph_idx = lens.asph_map[si]
            is_asph = asph_idx >= 0

            def trace_asph():
                coeffs = lens.asph_coeffs[asph_idx]
                K = coeffs[0]
                poly = coeffs[1:]
                _, ray, normal, defined = intersect_even_asphere(
                    c, K, poly, ray_local, ior=n_before
                )
                return ray, normal, defined

            ray_prop_local, normal, def_intersec = jx.cond(
                is_asph,
                trace_asph,
                trace_sph,
            )

        else:
            # We are dealing with a pure spherical lens.
            # So we avoid overhead of branching for aspheric surfaces.
            ray_prop_local, normal, def_intersec = trace_sph()

        # 3. Translate back to global coordinates.
        ray_prop = ray.translate(ray_prop_local, jnp.array([0.0, 0.0, z_surf]))

        # 4. Check aperture.
        r2_hit = vecop.vecnormsqr(ray_prop.o[..., 0:2])
        inside_aperture = r2_hit <= aperture**2

        pss_i = r2_hit - aperture**2

        # 5. Refract using Snell's law. (Undefined for total internal reflection)
        eta = n_before / n_after
        refract_dir, refraction_valid = refract(eta, normal, ray_prop.d)

        # 6. Update ray with refracted direction and block invalid rays.
        ray_next = ray_prop.set_d(refract_dir)
        ray_next = ray_next.block(def_intersec & inside_aperture & refraction_valid)

        if ray_next._pss is not None:
            ray_next = ray_next.set_pss(jnp.maximum(ray_next._pss, pss_i))

        return ray_next

    ray_out = jx.fori_loop(1, n_surfaces, trace_single_surface, input_ray)

    return ray_out


def load_tabular_lens_from_mytable(filepath: str) -> TabularLens:
    """
    Load a TabularLens from a mytable file.

    Args:
        filepath: Path to .mytable file
    """
    import pandas as pd

    from datasyn.optics.parse_mytable import parse_tables

    _MM2M = 1e-3  # Convert mm to m
    _M2MM = 1e3

    with open(filepath, "r") as f:
        text = f.read()
    tables = parse_tables(text)

    if "SURFACES" not in tables:
        raise ValueError(f"SURFACES table not found in {filepath}")

    df_surf = tables["SURFACES"]

    # Get image radius from last row
    if "APERDIAM" in df_surf.keys():
        imgrad = _MM2M * 0.5 * df_surf.iloc[-1]["APERDIAM"]
    else:
        imgrad = _MM2M * df_surf.iloc[-1]["APERSEMI"]

    # Remove last row (image plane)
    df_surf = df_surf.iloc[:-1].reset_index(drop=True)

    n_surfaces = len(df_surf)
    if n_surfaces == 0:
        raise ValueError("No surfaces found after filtering")

    # Parse ASPHERIC table if present
    df_asph = tables.get("ASPHERIC", None)
    if df_asph is not None and len(df_asph) > 0:
        # Get aspheric surface indices and coefficients
        asph_indices = df_asph["i"].values
        # Find coefficient columns (K, A4, A6, ...)
        coeff_cols = [c for c in df_asph.columns if c != "i"]
        asph_coeffs_list = df_asph[coeff_cols].values
    else:
        asph_indices = []
        asph_coeffs_list = None

    # Build surfaces array
    surfaces = []
    asph_map = []
    stop_idx = None
    z_cumsum = 0.0

    for idx, row in df_surf.iterrows():
        # Curvature
        R = row["R"]
        if pd.isna(R) or R is None:
            c = 0.0  # Flat surface
        else:
            c = 1.0 / (_MM2M * R)

        # Z position (cumulative)
        z = z_cumsum

        # Update z_cumsum for next surface
        D = row["D"]
        if idx != 0 and not pd.isna(D) and D is not None:
            z_cumsum += _MM2M * D

        # Aperture
        if "APERDIAM" in df_surf.keys():
            aper = 0.5 * row["APERDIAM"]
        else:
            aper = row["APERSEMI"]
        if pd.isna(aper) or aper is None:
            aper = float("inf")
        else:
            aper = _MM2M * aper

        # Cauchy coefficients
        Nd = row.get("Nd", None)
        Vd = row.get("Vd", None)
        if pd.isna(Nd) or Nd is None or pd.isna(Vd) or Vd is None:
            cauchy_A, cauchy_B = 1.0, 0.0  # Air
        else:
            cauchy_A, cauchy_B = get_cauchy_2term(Nd, Vd)

        surfaces.append([c, z, aper, cauchy_A, cauchy_B])

        # Aspheric mapping
        orig_idx = row["i"] if "i" in row else idx + 1
        if orig_idx in asph_indices:
            asph_row_idx = list(asph_indices).index(orig_idx)
            asph_map.append(asph_row_idx)
        else:
            asph_map.append(-1)

        # Stop detection
        if "STOP" in row and row["STOP"] == "X":
            stop_idx = idx

    # NOTE: This is for test now!
    imgpos = z_cumsum

    surfaces = jnp.array(surfaces, dtype=jnp.float64)
    asph_map = jnp.array(asph_map, dtype=jnp.int32)

    if asph_coeffs_list is not None and len(asph_coeffs_list) > 0:
        asph_coeffs = jnp.array(asph_coeffs_list, dtype=jnp.float64)
        # Note: coefficients in mytable are usually in mm units
        # K is dimensionless, but A4, A6, ... have units of 1/mm³, 1/mm⁵, ...
        # Need to convert: A_n [1/m^(n-1)] = A_n [1/mm^(n-1)] * (1000)^(n-1)
        # For A4: multiply by 1000^3 = 1e9
        # For A6: multiply by 1000^5 = 1e15
        # etc.
        n_cols = asph_coeffs.shape[1]
        if n_cols > 1:
            # First column is K (dimensionless), rest are A4, A6, ...
            scale_factors = jnp.array(
                [1.0] + [_M2MM ** (2 * i + 3) for i in range(n_cols - 1)]
            )
            asph_coeffs = asph_coeffs * scale_factors
    else:
        asph_coeffs = jnp.zeros((0, 1), dtype=jnp.float64)

    if stop_idx is None:
        stop_idx = 1  # Default to first surface if no stop marked

    return TabularLens(
        surfaces=surfaces,
        asph_map=asph_map,
        asph_coeffs=asph_coeffs,
        stop_idx=stop_idx,
        imgrad=imgrad,
        imgpos=imgpos,
    )
