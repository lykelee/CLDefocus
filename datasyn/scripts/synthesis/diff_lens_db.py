"""
Diff two lens SQLite tables (schema-agnostic).

Works for the lens `stats/data.db` and `classify/data.db` produced by
`datasyn.synthesis.lens.run` (both are SQLiteTableStore tables named `data`,
keyed by `name`), and for any other single-table SQLite DB with the same key.

Reports rows present in only one DB and rows present in both whose column
values differ. Use it to pin down how a lens collection diverged between two
runs (the confirmed cause of the reproducibility gap).

Examples::

    python scripts/synthesis/diff_lens_db.py <A>/stats/data.db <B>/stats/data.db
    python scripts/synthesis/diff_lens_db.py <A>/classify/data.db <B>/classify/data.db
    python scripts/synthesis/diff_lens_db.py A.db B.db --tol 1e-9 --cols valid crit efl
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def _pick_table(con: sqlite3.Connection, override: str | None) -> str:
    if override:
        return override
    tables = [
        r[0]
        for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    ]
    if "data" in tables:
        return "data"
    if len(tables) == 1:
        return tables[0]
    raise SystemExit(f"ambiguous tables {tables}; pass --table")


def _read(path: Path, table: str | None, key: str) -> tuple[list[str], dict[str, dict]]:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        tbl = _pick_table(con, table)
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({tbl})").fetchall()]
        if key not in cols:
            raise SystemExit(f"{path}: key column {key!r} not in {cols}")
        rows = {
            r[key]: {c: r[c] for c in cols}
            for r in con.execute(f"SELECT * FROM {tbl}").fetchall()
        }
        return cols, rows
    finally:
        con.close()


def _equal(x, y, tol: float) -> bool:
    if x == y:
        return True
    if tol > 0 and isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return abs(float(x) - float(y)) <= tol
    return False


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diff two lens SQLite tables.")
    p.add_argument("a", type=Path, help="first db (baseline)")
    p.add_argument("b", type=Path, help="second db")
    p.add_argument("--table", default=None, help="table name (default: auto / 'data')")
    p.add_argument("--key", default="name", help="primary-key column (default: name)")
    p.add_argument(
        "--tol",
        type=float,
        default=0.0,
        help="numeric tolerance for value equality (default: 0 = exact)",
    )
    p.add_argument(
        "--cols",
        nargs="*",
        default=None,
        help="restrict comparison to these columns (default: all shared)",
    )
    p.add_argument(
        "--stem-cols",
        nargs="*",
        default=["file"],
        help="columns compared by path stem, not full value (default: file); "
        "pass with no names to disable",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=50,
        help="max rows to print per section (default: 50; 0 = unlimited)",
    )
    return p.parse_args()


def _cap(items: list, limit: int) -> list:
    return items if limit <= 0 else items[:limit]


def main() -> int:
    args = _parse_args()
    cols_a, rows_a = _read(args.a, args.table, args.key)
    cols_b, rows_b = _read(args.b, args.table, args.key)

    keys_a, keys_b = set(rows_a), set(rows_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    common = sorted(keys_a & keys_b)

    shared_cols = [c for c in cols_a if c in cols_b and c != args.key]
    if args.cols:
        shared_cols = [c for c in shared_cols if c in args.cols]

    stem_cols = set(args.stem_cols or [])

    def _cmp_val(c, v):
        return Path(str(v)).stem if c in stem_cols else v

    changed: list[tuple[str, list[str]]] = []
    for k in common:
        ra, rb = rows_a[k], rows_b[k]
        diffs = []
        for c in shared_cols:
            va, vb = _cmp_val(c, ra[c]), _cmp_val(c, rb[c])
            if not _equal(va, vb, args.tol):
                diffs.append(f"{c}: {va!r} -> {vb!r}")
        if diffs:
            changed.append((k, diffs))

    if cols_a != cols_b:
        print(f"NOTE: column sets differ\n  A: {cols_a}\n  B: {cols_b}\n")

    print(f"== only in A ({len(only_a)}) ==")
    for k in _cap(only_a, args.limit):
        print(f"  {k}")
    print(f"== only in B ({len(only_b)}) ==")
    for k in _cap(only_b, args.limit):
        print(f"  {k}")
    print(f"== changed ({len(changed)}) ==")
    for k, diffs in _cap(changed, args.limit):
        print(f"  {k}  " + " | ".join(diffs))

    print(
        f"\nsummary: A={len(rows_a)} B={len(rows_b)} common={len(common)} "
        f"only_a={len(only_a)} only_b={len(only_b)} changed={len(changed)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
