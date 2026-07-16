"""
Dump / compare a Stage-3 (s3_camera) CameraStore.

CameraStore is a single SQLite file (`cameras.db`) holding one scalar row per
patch: chosen `lens`, `focusing` distance, and `sgncoc_max`. This tool prints
those rows, and can diff two stores row-by-row to see exactly which patches got
a different camera assignment between two runs (useful for reproducibility
debugging: a different lens/focusing here changes the signed-CoC depth
clustering in s4_patch_gen).

Patches are selected by any mix of:
  - integer index into the sorted patch-name list (negatives allowed), e.g., `0`, `-1`
  - index range `a:b` (b exclusive; `a`/`b` may be empty), e.g., `10:20`, `:5`, `900:`
  - literal patch id, e.g., `c000__US02250337-1__0003`
With no selectors, all rows are used.

Examples::

    python scripts/synthesis/dump_camera_store.py --store <root>/s3_camera_choose/cameras.db 0:20
    python scripts/synthesis/dump_camera_store.py \
        --store <A>/s3_camera_choose/cameras.db \
        --compare <B>/s3_camera_choose/cameras.db
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _is_int(tok: str) -> bool:
    return tok.lstrip("-").isdigit()


def _parse_range(tok: str, n: int) -> range | None:
    if ":" not in tok:
        return None
    parts = tok.split(":")
    if len(parts) != 2 or not all(p == "" or _is_int(p) for p in parts):
        return None
    a = int(parts[0]) if parts[0] else 0
    b = int(parts[1]) if parts[1] else n
    if a < 0:
        a += n
    if b < 0:
        b += n
    a = max(0, min(a, n))
    b = max(0, min(b, n))
    return range(a, b)


def _resolve(tokens: list[str], names: list[str]) -> list[str]:
    """Resolve selector tokens to an ordered, de-duplicated list of patch names."""
    n = len(names)
    out: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        if name not in seen:
            seen.add(name)
            out.append(name)

    for tok in tokens:
        tok = tok.strip()
        rng = _parse_range(tok, n)
        if rng is not None:
            for i in rng:
                add(names[i])
        elif _is_int(tok):
            i = int(tok)
            if i < 0:
                i += n
            if not 0 <= i < n:
                raise SystemExit(f"index out of range [0,{n}): {tok}")
            add(names[i])
        else:
            add(tok)
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dump / compare a CameraStore.")
    p.add_argument(
        "selectors",
        nargs="*",
        help="patch ids, integer indices, and/or index ranges a:b (default: all)",
    )
    p.add_argument(
        "--store",
        type=Path,
        required=True,
        help="CameraStore db (e.g., <data.root>/s3_camera_choose/cameras.db)",
    )
    p.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="second CameraStore db; print only rows that differ from --store",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="also write selected rows to this CSV path",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    from datasyn.synthesis.common import IOMode
    from datasyn.synthesis.stores.camera_store import CameraStore

    store = CameraStore(args.store, mode=IOMode.READ)
    names = store.get_all_names()
    selected = _resolve(args.selectors, names) if args.selectors else names

    if args.compare is not None:
        other = CameraStore(args.compare, mode=IOMode.READ)
        other_names = set(other.get_all_names())
        n_diff = 0
        for nm in selected:
            a = store.load(nm)
            if nm not in other_names:
                print(f"MISSING-in-compare {nm}")
                n_diff += 1
                continue
            b = other.load(nm)
            if (a.lens, a.focusing, a.sgncoc_max) != (b.lens, b.focusing, b.sgncoc_max):
                n_diff += 1
                print(
                    f"DIFF {nm}\n"
                    f"    lens:    {a.lens!r} -> {b.lens!r}\n"
                    f"    focusing:{a.focusing!r} -> {b.focusing!r}\n"
                    f"    sgncoc:  {a.sgncoc_max!r} -> {b.sgncoc_max!r}"
                )
        print(f"compared {len(selected)} patches: {n_diff} differ")
        return 0

    name_to_idx = {nm: i for i, nm in enumerate(names)}
    rows = []
    for i, nm in enumerate(selected):
        c = store.load(nm)
        idx = name_to_idx.get(nm, i)
        print(
            f"{idx}\t{nm}\tlens={c.lens}\t"
            f"focusing={c.focusing:.6g}\tsgncoc_max={c.sgncoc_max:.6g}"
        )
        rows.append((idx, nm, c.lens, c.focusing, c.sgncoc_max))

    if args.csv is not None:
        import csv

        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index", "name", "lens", "focusing", "sgncoc_max"])
            w.writerows(rows)
        print(f"wrote {len(rows)} rows -> {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
