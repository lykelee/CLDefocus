"""
Dump patches from a Stage-1 (s1_patch_crop) PatchStore to PNGs for inspection.

The PatchStore keeps linear/depth patches in LMDB + a SQLite definition table,
so the contents are not directly viewable. This tool renders selected patches
to PNG.

Patches are selected by any mix of:
  - integer index into the sorted patch-name list (negatives allowed), e.g., `0`, `-1`
  - index range `a:b` (b exclusive; `a`/`b` may be empty), e.g., `10:20`, `:5`, `900:`
  - literal patch id, e.g., `c000__US02250337-1_Example__0003`

For each patch it writes `<name>__lin.png` (linear patch, gamma-encoded for
display) and `<name>__dpt.png` (turbo inverse-depth colormap, identical to the
pipeline's depth viz via `mldepthpro_depth2rgb`).

Examples::

    python scripts/synthesis/viz_patch_store.py \
        --store <data.root>/s1_patch_crop/patches 0 5 10:20 c000__PHOTO__0003
    python scripts/synthesis/viz_patch_store.py --store <patches> --list
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


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
            add(tok)  # literal patch id; validated at load time
    return out


def _linear_to_png_u8(lin_u16) -> np.ndarray:
    x = np.clip(np.asarray(lin_u16, dtype=np.float32) / 65535.0, 0.0, 1.0)
    return (x ** (1 / 2.2) * 255.0).astype(np.uint8)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render PatchStore patches to PNG.")
    p.add_argument(
        "selectors",
        nargs="*",
        help="patch ids, integer indices, and/or index ranges a:b",
    )
    p.add_argument(
        "--store",
        type=Path,
        required=True,
        help="PatchStore directory (e.g., <data.root>/s1_patch_crop/patches)",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output directory for PNGs",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="print total count and all names (index<TAB>name), then exit",
    )
    p.add_argument("--no-linear", action="store_true", help="skip linear PNGs")
    p.add_argument("--no-depth", action="store_true", help="skip depth PNGs")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    from datasyn.synthesis.common import IOMode
    from datasyn.synthesis.stores.patch_store import PatchStore

    store = PatchStore(args.store, mode=IOMode.READ)
    names = store.get_all_names()

    if args.list or not args.selectors:
        print(f"{len(names)} patches in {args.store}")
        for i, nm in enumerate(names):
            print(f"{i}\t{nm}")
        return 0

    selected = _resolve(args.selectors, names)
    args.out.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    saved = 0
    for nm in selected:
        try:
            patdef = store.load_patdef(nm)
        except KeyError:
            print(f"SKIP unknown patch id: {nm}")
            continue

        if not args.no_linear:
            lin = store.load_linear(nm)
            Image.fromarray(_linear_to_png_u8(lin)).save(args.out / f"{nm}__lin.png")

        if not args.no_depth:
            from datasyn.synthesis.patch_define import mldepthpro_depth2rgb

            dpt = np.asarray(store.load_depth(nm), dtype=np.float32)
            rgb = np.asarray(mldepthpro_depth2rgb(dpt))
            Image.fromarray(rgb).save(args.out / f"{nm}__dpt.png")

        saved += 1
        print(
            f"saved {nm}  photo={patdef.photo} "
            f"depth=[{patdef.depth_min:.3f}, {patdef.depth_max:.3f}]"
        )

    print(f"done: {saved}/{len(selected)} patches -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
