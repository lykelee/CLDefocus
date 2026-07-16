"""
Verify lens-pool partitioning across `lens_pool.txt` files of camera choose stage.
Aim to use to validate train/validation/test splits.

Usage:
    python verify_lens_pools.py train/lens_pool.txt val/lens_pool.txt test/lens_pool.txt
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path


def load_pool(path: str) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def main(argv: list[str]) -> int:
    paths = argv[1:]
    if len(paths) < 1:
        print("usage: verify_lens_pools.py POOL1.txt [POOL2.txt ...]", file=sys.stderr)
        return 2

    pools: dict[str, list[str]] = {}
    sets: dict[str, set[str]] = {}
    for p in paths:
        names = load_pool(p)
        pools[p] = names
        sets[p] = set(names)

    print("=" * 60)
    print("Per-pool lens counts")
    print("=" * 60)
    for p in paths:
        names = pools[p]
        n_total = len(names)
        n_distinct = len(sets[p])
        dup_note = (
            ""
            if n_total == n_distinct
            else f"  (!) {n_total - n_distinct} dup within file"
        )
        print(f"  {n_distinct:5d} distinct  |  {p}{dup_note}")

    print()
    print("=" * 60)
    print("Pairwise overlap (leakage check)")
    print("=" * 60)
    any_overlap = False
    if len(paths) < 2:
        print("  (single pool — nothing to compare)")
    else:
        for a, b in combinations(paths, 2):
            inter = sets[a] & sets[b]
            if inter:
                any_overlap = True
                sample = ", ".join(sorted(inter)[:5])
                more = " ..." if len(inter) > 5 else ""
                print(
                    f"  OVERLAP {len(inter):5d}  between\n      {a}\n      {b}\n      e.g., {sample}{more}"
                )
            else:
                print(f"  ok       0     {a}  vs  {b}")

    union = set()
    for s in sets.values():
        union |= s
    sum_distinct = sum(len(s) for s in sets.values())

    print()
    print("=" * 60)
    print("Totals")
    print("=" * 60)
    print(f"  distinct lenses (union)      : {len(union)}")
    print(f"  sum of per-pool distinct     : {sum_distinct}")
    print(
        f"  difference (sum - union)     : {sum_distinct - len(union)}  (0 == perfectly disjoint)"
    )
    print()
    if any_overlap:
        print("RESULT: pools OVERLAP — lens leakage between splits.")
        return 1
    print("RESULT: pools are disjoint — no lens leakage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
