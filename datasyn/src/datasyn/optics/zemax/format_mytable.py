"""
mytable formatter: Convert pandas DataFrames to mytable text format.

This is the inverse of parse_mytable.parse_tables().

mytable format:
- Table name on its own line (unindented)
- Header row (indented, columns separated by 2+ spaces)
- Optional rule line (dashes)
- Data rows (indented, columns separated by 2+ spaces)

Cell value encoding:
- None -> "-"
- float("inf") -> "INF"
- int -> as-is
- float -> general format, scientific for extreme values
- str -> as-is
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import pandas as pd

# =============================================================================
# Cell formatting
# =============================================================================


def _format_cell(value: Any, float_precision: int = 5) -> str:
    """
    Convert a Python value to mytable cell text.

    Args:
        value: The value to format
        float_precision: Decimal places for float formatting

    Returns:
        String representation for mytable
    """
    if value is None:
        return "-"

    if isinstance(value, float):
        if math.isinf(value):
            return "INF"
        if math.isnan(value):
            return "-"
        # Use scientific notation for very small or large values
        abs_val = abs(value)
        if abs_val == 0.0:
            return "0"
        if abs_val < 1e-4 or abs_val >= 1e6:
            return f"{value:.{float_precision}E}"
        # General format - strip trailing zeros
        formatted = f"{value:.{float_precision}f}".rstrip("0").rstrip(".")
        return formatted

    if isinstance(value, bool):
        # Must check before int since bool is subclass of int
        return "X" if value else "-"

    if isinstance(value, int):
        return str(value)

    # String or other - convert to string
    return str(value)


def _is_numeric_column(df: pd.DataFrame, col: str) -> bool:
    """Check if a column contains numeric values (for alignment)."""
    for val in df[col]:
        if val is not None and not isinstance(val, (int, float)):
            return False
    return True


# =============================================================================
# Table formatting
# =============================================================================


def format_table(
    name: str,
    df: pd.DataFrame,
    *,
    indent: str = "  ",
    min_col_gap: int = 2,
    float_precision: int = 5,
    include_rule: bool = True,
) -> str:
    """
    Format a single DataFrame as mytable table text.

    Args:
        name: Table name
        df: DataFrame to format
        indent: Indentation string for header and data rows
        min_col_gap: Minimum spaces between columns
        float_precision: Decimal places for float values
        include_rule: Whether to include a rule line after header

    Returns:
        Formatted table as string
    """
    if df.empty:
        # Empty table - just name and header
        header_line = indent + ((" " * min_col_gap).join(df.columns))
        return f"{name}\n{header_line}\n"

    columns = list(df.columns)
    n_cols = len(columns)

    # Format all cells
    formatted_cells: List[List[str]] = []
    for _, row in df.iterrows():
        formatted_row = [_format_cell(row[col], float_precision) for col in columns]
        formatted_cells.append(formatted_row)

    # Calculate column widths (max of header and all cells)
    col_widths = []
    for i, col in enumerate(columns):
        header_width = len(str(col))
        cell_widths = [len(formatted_cells[r][i]) for r in range(len(formatted_cells))]
        max_width = max([header_width] + cell_widths)
        col_widths.append(max_width)

    # Determine alignment for each column (right for numeric, left for string)
    col_alignments = []
    for col in columns:
        if _is_numeric_column(df, col):
            col_alignments.append(">")  # Right align
        else:
            col_alignments.append("<")  # Left align

    # Build output
    lines = [name]

    # Header line
    header_parts = []
    for i, col in enumerate(columns):
        # Headers are typically left-aligned or centered
        header_parts.append(f"{col:<{col_widths[i]}}")
    header_line = indent + (" " * min_col_gap).join(header_parts)
    lines.append(header_line)

    # Rule line
    if include_rule:
        rule_width = len(header_line) - len(indent) + 10  # A bit extra
        rule_line = indent + "-" * rule_width
        lines.append(rule_line)

    # Data rows
    for row_cells in formatted_cells:
        row_parts = []
        for i, cell in enumerate(row_cells):
            align = col_alignments[i]
            row_parts.append(f"{cell:{align}{col_widths[i]}}")
        row_line = indent + (" " * min_col_gap).join(row_parts)
        lines.append(row_line)

    return "\n".join(lines) + "\n"


def format_tables(
    tables: Dict[str, pd.DataFrame],
    *,
    indent: str = "  ",
    min_col_gap: int = 2,
    float_precision: int = 5,
    include_rule: bool = True,
) -> str:
    """
    Format multiple DataFrames as mytable text.

    Args:
        tables: Dictionary mapping table names to DataFrames
        indent: Indentation string for header and data rows
        min_col_gap: Minimum spaces between columns
        float_precision: Decimal places for float values
        include_rule: Whether to include rule lines after headers

    Returns:
        Complete mytable text with all tables
    """
    parts = []
    for name, df in tables.items():
        table_text = format_table(
            name,
            df,
            indent=indent,
            min_col_gap=min_col_gap,
            float_precision=float_precision,
            include_rule=include_rule,
        )
        parts.append(table_text)

    return "\n".join(parts)


# =============================================================================
# Demo / Test
# =============================================================================


def demo():
    """Demo: Format some test data and verify round-trip with parser."""
    import numpy as np

    from datasyn.optics.parse_mytable import parse_tables

    # Create test DataFrames
    surfaces_df = pd.DataFrame(
        {
            "i": [0, 1, 2, 3],
            "STOP": [None, None, "X", None],
            "SHAPE": ["FLAT", "SPHR", "ASPH", "FLAT"],
            "R": [None, 10.5, -5.25, None],
            "D": [float("inf"), 1.5, 0.8, None],
            "Nd": [None, 1.517, 1.623, None],
            "Vd": [None, 64.2, 58.1, None],
            "APERSEMI": [None, 2.5, 2.3, 1.8],
        }
    )

    aspheric_df = pd.DataFrame(
        {
            "i": [2],
            "K": [-1.5],
            "A4": [1.23e-4],
            "A6": [-5.67e-7],
            "A8": [0.0],
        }
    )

    tables = {"SURFACES": surfaces_df, "ASPHERIC": aspheric_df}

    print("=" * 60)
    print("Original DataFrames:")
    print("=" * 60)
    print("\nSURFACES:")
    print(surfaces_df)
    print("\nASPHERIC:")
    print(aspheric_df)

    print("\n" + "=" * 60)
    print("Formatted mytable text:")
    print("=" * 60)
    text = format_tables(tables)
    print(text)

    print("=" * 60)
    print("Round-trip test (parse back):")
    print("=" * 60)
    parsed = parse_tables(text)
    print("\nParsed SURFACES:")
    print(parsed["SURFACES"])
    print("\nParsed ASPHERIC:")
    print(parsed["ASPHERIC"])

    # Verify round-trip
    print("\n" + "=" * 60)
    print("Verification:")
    print("=" * 60)

    # Check that parsed values match original (allowing for formatting differences)
    for col in surfaces_df.columns:
        for i in range(len(surfaces_df)):
            orig = surfaces_df.iloc[i][col]
            parsed_val = parsed["SURFACES"].iloc[i][col]
            if orig is None and parsed_val is None:
                continue
            if isinstance(orig, float) and isinstance(parsed_val, float):
                if math.isnan(orig) and math.isnan(parsed_val):
                    continue
                if math.isinf(orig) and math.isinf(parsed_val):
                    continue
                if not np.isclose(orig, parsed_val, rtol=1e-4):
                    print(f"MISMATCH at SURFACES[{i}][{col}]: {orig} vs {parsed_val}")
            elif orig != parsed_val:
                print(f"MISMATCH at SURFACES[{i}][{col}]: {orig!r} vs {parsed_val!r}")

    print("Round-trip verification complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(demo())
