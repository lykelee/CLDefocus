"""
Convert Zemax ZMX files to mytable (tabular form) format.

This module directly parses ZMX files using the Stanza parser and extracts
only the information needed for mytable format:
- Surface geometry (curvature, thickness, aperture)
- Material properties (Nd, Vd via glass catalog lookup)
- Aspheric coefficients (conic constant and even polynomial terms)

Supported surface types:
- STANDARD (spherical or flat)
- EVENASPH (even polynomial aspheric)

Unsupported surface types will raise UnsupportedSurfaceTypeError.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from datasyn.optics.zemax.format_mytable import format_tables
from datasyn.optics.zemax.glass import NdVdGlassCatalog
from datasyn.optics.zemax.stanza import Stanza, parse_stanza_file

# =============================================================================
# Exceptions
# =============================================================================


class ZmxConversionError(Exception):
    """Base exception for ZMX conversion errors."""

    pass


class UnsupportedSurfaceTypeError(ZmxConversionError):
    """Raised when ZMX contains unsupported surface types."""

    def __init__(self, surf_num: int, surf_type: str):
        self.surf_num = surf_num
        self.surf_type = surf_type
        super().__init__(
            f"Surface {surf_num}: unsupported type '{surf_type}'. "
            f"Only STANDARD, EVENASPH and XASPHERE are supported."
        )


class UnknownGlassError(ZmxConversionError):
    """Raised when glass material cannot be resolved to Nd/Vd."""

    def __init__(self, surf_num: int, glass_name: str):
        self.surf_num = surf_num
        self.glass_name = glass_name
        super().__init__(
            f"Surface {surf_num}: unknown glass '{glass_name}' not found in catalog."
        )


class UnsupportedUnitError(ZmxConversionError):
    """Raised for unsupported unit systems."""

    def __init__(self, unit: str):
        self.unit = unit
        super().__init__(
            f"Unsupported unit '{unit}'. Only MM, CM, IN, M are supported."
        )


# =============================================================================
# Internal data structures
# =============================================================================


@dataclass
class _SurfaceData:
    """Internal representation of a surface for conversion."""

    index: int  # Original ZMX surface number
    is_stop: bool
    surf_type: str  # STANDARD or EVENASPH
    curv: float  # Curvature (1/R)
    thickness: float  # Distance to next surface
    glass_name: str | None  # None for air
    Nd: float | None
    Vd: float | None
    semi_aperture: float
    # Aspheric data (only for EVENASPH)
    conic: float  # K
    asph_coeffs: list[float]  # [A4, A6, A8, ...] from PARM 2, 3, 4, ...


# =============================================================================
# Unit conversion
# =============================================================================

_UNIT_TO_MM = {
    "MM": 1.0,
    # "CM": 10.0,
    # "IN": 25.4,
    # "INCH": 25.4,
    # "M": 1000.0,
}


def _get_unit_scale(unit_str: str) -> float:
    """Get scale factor to convert from given unit to mm."""
    unit_upper = unit_str.upper()
    if unit_upper not in _UNIT_TO_MM:
        raise UnsupportedUnitError(unit_str)
    return _UNIT_TO_MM[unit_upper]


# =============================================================================
# ZMX parsing helpers
# =============================================================================


def _parse_surface(
    surf: Stanza,
    glass_catalog: NdVdGlassCatalog,
    unit_scale: float,
) -> _SurfaceData:
    """Parse a single SURF stanza into _SurfaceData."""
    surf_num = surf.get_param_int(0, 0)

    # Surface type
    type_stanza = surf.get("TYPE")
    surf_type = type_stanza.params[0] if type_stanza else "STANDARD"

    # Validate surface type
    if surf_type not in ("STANDARD", "EVENASPH", "XASPHERE"):
        raise UnsupportedSurfaceTypeError(surf_num, surf_type)

    # Stop surface
    is_stop = surf.has("STOP")

    # Curvature
    curv_stanza = surf.get("CURV")
    curv = curv_stanza.get_param_float(0, 0.0) if curv_stanza else 0.0
    # Curvature is 1/R, and R scales with unit, so curv scales inversely
    curv = curv / unit_scale

    # Thickness (distance to next surface)
    disz_stanza = surf.get("DISZ")
    if disz_stanza:
        thickness = disz_stanza.get_param_float(0, 0.0)
        if thickness == float("inf"):
            thickness = float("inf")
        else:
            thickness = thickness * unit_scale
    else:
        thickness = 0.0

    # Glass material
    glas_stanza = surf.get("GLAS")
    glass_name = None
    Nd = None
    Vd = None

    if glas_stanza and glas_stanza.params:
        glass_name = glas_stanza.params[0]

        if glass_name == "MIRROR":
            raise ValueError("MIRROR is not supported yet!")
        elif glass_name == "___BLANK":
            # Inline Nd/Vd specification
            # Format: ___BLANK status nd vd ...
            Nd = glas_stanza.get_param_float(3)
            Vd = glas_stanza.get_param_float(4)
            if Nd is None or Vd is None:
                raise ZmxConversionError(
                    f"Surface {surf_num}: ___BLANK glass missing Nd/Vd values"
                )
        else:
            # Lookup in catalog
            glass_entry = glass_catalog[glass_name]
            if glass_entry is None:
                raise UnknownGlassError(surf_num, glass_name)
            Nd, Vd = glass_entry.Nd_Vd

    # Semi-aperture (diameter)
    diam_stanza = surf.get("DIAM")
    semi_aperture = diam_stanza.get_param_float(0, 0.0) if diam_stanza else 0.0
    semi_aperture = semi_aperture * unit_scale

    # Conic constant (for aspheric surfaces)
    coni_stanza = surf.get("CONI")
    conic = coni_stanza.get_param_float(0, 0.0) if coni_stanza else 0.0

    # Aspheric polynomial coefficients
    # PARM 1 is typically 0 (or r^2 term which is redundant with curvature)
    # PARM 2 = A4, PARM 3 = A6, PARM 4 = A8, etc.
    asph_coeffs = []
    if surf_type == "EVENASPH":
        parm_stanzas = surf["PARM"]
        # Collect PARM 2 through PARM 8 (A4 through A16)
        parm_dict = {}
        for parm in parm_stanzas:
            idx = parm.get_param_int(0)
            val = parm.get_param_float(1, 0.0)
            if idx is not None and idx >= 2:
                parm_dict[idx] = val

        # Build coefficient list [A4, A6, A8, ...]
        # PARM indices 2, 3, 4, ... correspond to A4, A6, A8, ...
        max_parm = max(parm_dict.keys()) if parm_dict else 1
        for i in range(2, max_parm + 1):
            coeff = parm_dict.get(i, 0.0)
            # Scale coefficient: A_n has units of 1/length^(n-1)
            # A4 has units 1/mm^3, A6 has units 1/mm^5, etc.
            # n = 2 * i, so power = n - 1 = 2*i - 1
            power = 2 * i - 1
            coeff = coeff / (unit_scale**power)
            asph_coeffs.append(coeff)

    elif surf_type == "XASPHERE":
        xdat_stanzas = surf["XDAT"]

        # Collect XDAT 4, XDAT 5, ...,
        xdat_dict = {}
        for xdat in xdat_stanzas:
            idx = xdat.get_param_int(0)
            val = xdat.get_param_float(1, 0.0)
            if idx is not None and idx >= 4:
                xdat_dict[idx] = val

        # XDAT indices 4, 5, 6, ... correspond to A4, A6, A8, ...
        max_xdat = max(xdat_dict.keys()) if xdat_dict else 1
        for i in range(4, max_xdat + 1):
            coeff = xdat_dict.get(i, 0.0)
            # Scale coefficient: A_n has units of 1/length^(n-1)
            # A4 has units 1/mm^3, A6 has units 1/mm^5, etc.
            # n = 2 * i, so power = n - 1 = 2*i - 1
            power = 2 * i - 5
            coeff = coeff / (unit_scale**power)
            asph_coeffs.append(coeff)

    return _SurfaceData(
        index=surf_num,
        is_stop=is_stop,
        surf_type=surf_type,
        curv=curv,
        thickness=thickness,
        glass_name=glass_name,
        Nd=Nd,
        Vd=Vd,
        semi_aperture=semi_aperture,
        conic=conic,
        asph_coeffs=asph_coeffs,
    )


# =============================================================================
# Build DataFrames from parsed surface data
# =============================================================================


def _build_dataframes(
    surfaces: list[_SurfaceData],
    max_asph_order: int,
) -> dict[str, pd.DataFrame]:
    """
    Build SURFACES and ASPHERIC DataFrames from parsed surface data.

    Args:
        surfaces: List of parsed surface data
        max_asph_order: Number of aspheric coefficients (A4, A6, ...)

    Returns:
        Dictionary with "SURFACES" and optionally "ASPHERIC" DataFrames
    """
    n_surfaces = len(surfaces)

    # ==========================================================================
    # SURFACES DataFrame
    # ==========================================================================
    surf_rows = []
    for i, surf in enumerate(surfaces):
        is_last = i == n_surfaces - 1

        # STOP: "X" or None
        stop = "X" if surf.is_stop else None

        # SHAPE: FLAT, SPHR, or ASPH
        if surf.surf_type in ["EVENASPH", "XASPHERE"]:
            shape = "ASPH"
        elif surf.curv == 0.0:
            shape = "FLAT"
        else:
            shape = "SPHR"

        # R (radius of curvature)
        R = None if surf.curv == 0.0 else 1.0 / surf.curv

        # D (thickness) - last surface has None
        D = None if is_last else surf.thickness

        # Nd, Vd
        Nd = surf.Nd
        Vd = surf.Vd

        # APERSEMI - None if zero (like object plane)
        apersemi = None if surf.semi_aperture == 0.0 else surf.semi_aperture

        surf_rows.append(
            {
                "i": i,
                "STOP": stop,
                "SHAPE": shape,
                "R": R,
                "D": D,
                "Nd": Nd,
                "Vd": Vd,
                "APERSEMI": apersemi,
            }
        )

    surfaces_df = pd.DataFrame(surf_rows)

    # ==========================================================================
    # ASPHERIC DataFrame (only if there are aspheric surfaces)
    # ==========================================================================
    asph_rows = []
    for i, surf in enumerate(surfaces):
        if surf.surf_type not in ["EVENASPH", "XASPHERE"]:
            continue

        row = {"i": i, "K": surf.conic}

        # Add coefficients A4, A6, A8, ...
        coeffs = surf.asph_coeffs
        for j in range(max_asph_order):
            col_name = f"A{2 * (j + 2)}"  # A4, A6, A8, ...
            row[col_name] = coeffs[j] if j < len(coeffs) else 0.0

        asph_rows.append(row)

    tables = {"SURFACES": surfaces_df}

    if asph_rows:
        aspheric_df = pd.DataFrame(asph_rows)
        tables["ASPHERIC"] = aspheric_df

    return tables


# =============================================================================
# Main conversion function
# =============================================================================


def zmx_to_mytable(
    zmx_path: str | Path,
    glass_catalog: NdVdGlassCatalog,
) -> str:
    """
    Convert Zemax ZMX file to mytable format.

    Args:
        zmx_path: Path to the ZMX file
        glass_catalog: Glass catalog for Nd/Vd lookup

    Returns:
        mytable format as string

    Raises:
        UnsupportedSurfaceTypeError: If ZMX contains non-spherical/aspheric surfaces
        UnknownGlassError: If glass not found in catalog
        UnsupportedUnitError: If unsupported unit system
    """
    # Parse ZMX file
    doc = parse_stanza_file(zmx_path)

    # Get unit scale
    unit_stanza = doc.get("UNIT")
    unit_str = unit_stanza.params[0] if unit_stanza else "MM"
    unit_scale = _get_unit_scale(unit_str)

    # Parse all surfaces
    surf_stanzas = doc["SURF"]
    surfaces: list[_SurfaceData] = []

    for surf in surf_stanzas:
        surf_data = _parse_surface(surf, glass_catalog, unit_scale)
        surfaces.append(surf_data)

    # Keep all surfaces including object plane (surface 0) and image plane (last)
    # The mytable format expects:
    # - Surface 0: object plane with D=INF
    # - Surfaces 1 to N-1: actual lens surfaces
    # - Surface N: image plane with D="-"

    # Determine max aspheric order from all surfaces
    max_asph_order = 0
    for surf in surfaces:
        if surf.asph_coeffs:
            max_asph_order = max(max_asph_order, len(surf.asph_coeffs))

    # Default to at least A4-A16 (7 coefficients) if any aspheric
    if max_asph_order > 0:
        max_asph_order = max(max_asph_order, 7)

    # Build DataFrames
    tables = _build_dataframes(surfaces, max_asph_order)

    # Format as mytable text
    return format_tables(tables, float_precision=8)


def zmx_to_mytable_file(
    zmx_path: str | Path,
    output_path: str | Path,
    glass_catalog: NdVdGlassCatalog,
):
    """
    Convert Zemax ZMX file to mytable file.

    Args:
        zmx_path: Path to the ZMX file
        output_path: Path for output mytable file
        glass_catalog: Glass catalog for Nd/Vd lookup
    """
    mytable_str = zmx_to_mytable(zmx_path, glass_catalog)

    output_path = Path(output_path)
    output_path.write_text(mytable_str, encoding="utf-8")


# =============================================================================
# Demo / Test
# =============================================================================

# Demo is moved to `scripts/zemax/zmx_to_mytable.py`.
