from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List

import pandas as pd

# Split on 2+ spaces or tabs, preserving tokens like 'A10'
_SPLIT_RE = re.compile(r"(?:\t+|\s{2,})")


def _is_comment_line(s: str) -> bool:
    """Check if line is a comment (starts with #)."""
    return s.lstrip().startswith("#")


def _is_rule_line(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    # Accept common rule chars
    return set(s) <= set("-—_=.•+ ")


def _split_fields(line: str) -> List[str]:
    # Strip one level of indent, then split
    return [tok for tok in _SPLIT_RE.split(line.strip()) if tok != ""]


_NUM_RE = re.compile(r"^[+-]?((\d+(\.\d*)?)|(\.\d+))([eE][+-]?\d+)?$")


def _coerce_cell(tok: str) -> Any:
    if tok == "-":
        return None
    if tok.upper() == "INF":
        return float("inf")
    # ints first (no decimal/exponent)
    if tok.isdigit() or (tok.startswith(("-", "+")) and tok[1:].isdigit()):
        try:
            return int(tok)
        except ValueError:
            pass
    # floats (incl. scientific)
    if _NUM_RE.match(tok):
        try:
            return float(tok)
        except ValueError:
            pass
    return tok  # keep as string (e.g., "X", "FLAT", "ASPH")


def parse_tables(text: str) -> Dict[str, pd.DataFrame]:
    """
    Parse the custom multi-table format into DataFrames keyed by table name.
    """
    lines = [ln.rstrip("\n") for ln in text.splitlines()]

    tables: Dict[str, pd.DataFrame] = {}
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        # Skip comments, blank lines, and stray indented lines
        if line.strip() == "" or _is_comment_line(line):
            i += 1
            continue
        if line.startswith((" ", "\t")):
            # Skip stray indented lines outside a table (unlikely, but safe)
            i += 1
            continue

        table_name = line.strip()
        if not table_name:
            i += 1
            continue
        if table_name in tables:
            warnings.warn(
                f"Duplicate table name '{table_name}' - overwriting previous table"
            )
        i += 1

        # Skip blank lines and comments until header
        while i < n and (lines[i].strip() == "" or _is_comment_line(lines[i])):
            i += 1
        if i >= n:
            break

        # Header (indented)
        header_line = lines[i]
        if not header_line.startswith((" ", "\t")):
            raise ValueError(
                f"Expected header after table '{table_name}', got: {header_line!r}"
            )
        header = _split_fields(header_line)
        i += 1

        # Optional rule line
        if i < n and _is_rule_line(lines[i]):
            i += 1

        # Data rows: consecutive indented, non-empty lines
        rows: List[List[Any]] = []
        row_num = 0
        while i < n:
            row_line = lines[i]
            if row_line.strip() == "" or _is_comment_line(row_line):
                i += 1
                continue
            if not row_line.startswith((" ", "\t")):
                break  # next table
            if _is_rule_line(row_line):
                i += 1
                continue
            fields = _split_fields(row_line)
            # Warn on field count mismatch
            if len(fields) < len(header):
                warnings.warn(
                    f"Table '{table_name}' row {row_num}: expected {len(header)} fields, "
                    f"got {len(fields)} - padding with None"
                )
                fields += ["-"] * (len(header) - len(fields))
            elif len(fields) > len(header):
                warnings.warn(
                    f"Table '{table_name}' row {row_num}: expected {len(header)} fields, "
                    f"got {len(fields)} - truncating extras"
                )
                fields = fields[: len(header)]
            rows.append([_coerce_cell(tok) for tok in fields])
            row_num += 1
            i += 1

        if not rows:
            warnings.warn(f"Table '{table_name}' has no data rows")

        df = pd.DataFrame(rows, columns=header)
        tables[table_name] = df

    return tables
