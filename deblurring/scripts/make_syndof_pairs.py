"""
SYNDOF paired symlink creator.
Matches synthesized inputs to GT sources via filename parsing, resolves SYNTHIA
ambiguity by PSNR, splits into train/val, and writes input/target symlinks under
out_root/{train,val}/{input,target}.

Importable: setup_data.py calls make_syndof(...). Also runnable directly:
    python scripts/make_syndof_pairs.py \
        --gt-root /raw/syndof/source/synthetic_datasets/image \
        --input-root /raw/syndof/generated/train/SYNDOF/image \
        --out-root data/SYNDOF --train-size 8000 [--dry-run]
"""

import os
import re
import argparse
from pathlib import Path

import numpy as np
import cv2
from tqdm.auto import tqdm

DEFAULT_PSNR_THRESHOLD = 15.0  # dB; ambiguous SYNTHIA match must beat this while others fall below


def build_gt_lookup(gt_root: Path) -> dict:
    """Build key -> [Path, ...] lookup from all GT images."""
    lookup: dict[str, list[Path]] = {}

    for p in (gt_root / "middleburry").rglob("*.png"):
        # middleburry/2014/{scene}/im*.png  ->  key ignores year dir
        key = f"MIDDLEBURRY_{p.parent.name}_{p.stem}"
        lookup.setdefault(key, []).append(p)

    for p in (gt_root / "MPI").rglob("*.png"):
        # MPI/{seq}/{frame}.png
        key = f"MPI_{p.parent.name}_{p.stem}"
        lookup.setdefault(key, []).append(p)

    for p in (gt_root / "SYNTHIA").rglob("*.png"):
        # SYNTHIA/{SEQ}/Stereo_Right/{Omni_X}/{frame}.png ; SEQ missing from inputs
        key = f"SYNTHIA_{p.parent.name}_{p.stem}"
        lookup.setdefault(key, []).append(p)

    return lookup


def parse_key(filename: str) -> str | None:
    """Extract GT lookup key from input filename (strip type prefix + synth params)."""
    stem = Path(filename).stem
    for prefix in ("MIDDLEBURRY", "MPI", "SYNTHIA"):
        if stem.startswith(prefix + "_"):
            rest = stem[len(prefix) + 1 :]
            m = re.search(r"_f_\d", rest)
            gt_part = rest[: m.start()] if m else rest
            return f"{prefix}_{gt_part}"
    return None


def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0**2 / mse)


def resolve_ambiguous(
    input_path: Path, candidates: list[Path], psnr_threshold: float
) -> tuple[Path, bool]:
    """Pick best GT by PSNR vs input. is_confident = best >= threshold AND all others < threshold."""
    inp_img = cv2.imread(str(input_path))
    h, w = inp_img.shape[:2]

    psnrs = []
    for gt_path in candidates:
        gt_img = cv2.imread(str(gt_path))
        if gt_img is None:
            psnrs.append(-1.0)
            continue
        if gt_img.shape[:2] != (h, w):
            gt_img = cv2.resize(gt_img, (w, h), interpolation=cv2.INTER_LINEAR)
        psnrs.append(compute_psnr(inp_img, gt_img))

    best_idx = int(np.argmax(psnrs))
    best_psnr = psnrs[best_idx]
    other_psnrs = [p for i, p in enumerate(psnrs) if i != best_idx]
    confident = best_psnr >= psnr_threshold and all(p < psnr_threshold for p in other_psnrs)
    return candidates[best_idx], confident


def make_relative_symlink(target: Path, link: Path, dry_run: bool):
    rel = Path(os.path.relpath(target, link.parent))
    if not dry_run:
        link.symlink_to(rel)


def make_syndof(
    gt_root,
    input_root,
    out_root,
    train_size: int,
    dry_run: bool = False,
    psnr_threshold: float = DEFAULT_PSNR_THRESHOLD,
) -> dict:
    """Build SYNDOF train/val input/target symlinks under out_root. Returns stats dict."""
    gt_root, input_root, out_root = Path(gt_root), Path(input_root), Path(out_root)
    train_input = out_root / "train" / "input"
    train_target = out_root / "train" / "target"
    val_input = out_root / "val" / "input"
    val_target = out_root / "val" / "target"

    if dry_run:
        print("[DRY RUN] No files will be created.\n")

    print("Building GT lookup...")
    lookup = build_gt_lookup(gt_root)
    n_gt_imgs = sum(len(v) for v in lookup.values())
    n_dup_keys = sum(1 for v in lookup.values() if len(v) > 1)
    print(f"  {n_gt_imgs} GT images | {len(lookup)} unique keys | {n_dup_keys} ambiguous keys\n")

    input_files = sorted(input_root.glob("*.png"))
    stats = {
        "matched_unique": 0,
        "ambiguous_confident": 0,
        "ambiguous_uncertain": 0,
        "unmatched_no_key": 0,
        "unmatched_not_found": 0,
        "skipped_exists": 0,
    }
    uncertain_cases: list[dict] = []
    unmatched_cases: list[str] = []

    # Phase 1: resolve pairs
    pairs: list[tuple[Path, Path]] = []
    for inp in tqdm(input_files, desc="Resolving pairs", unit="img", dynamic_ncols=True):
        key = parse_key(inp.name)
        if key is None:
            stats["unmatched_no_key"] += 1
            unmatched_cases.append(f"[no_key] {inp.name}")
            continue
        candidates = lookup.get(key)
        if not candidates:
            stats["unmatched_not_found"] += 1
            unmatched_cases.append(f"[not_found key={key}] {inp.name}")
            continue
        if len(candidates) == 1:
            gt = candidates[0]
            stats["matched_unique"] += 1
        else:
            gt, confident = resolve_ambiguous(inp, candidates, psnr_threshold)
            if confident:
                stats["ambiguous_confident"] += 1
            else:
                stats["ambiguous_uncertain"] += 1
                uncertain_cases.append(
                    {
                        "input": inp.name,
                        "chosen_gt": str(gt),
                        "candidates": [str(c) for c in candidates],
                    }
                )
        pairs.append((inp, gt))

    # Phase 2: shuffle & split (fixed seed for reproducibility)
    rng = np.random.default_rng(seed=0)
    order = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in order]
    if train_size > len(pairs):
        raise ValueError(f"train_size {train_size} > total pairs {len(pairs)}")
    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:]

    # Phase 3: symlinks
    if not dry_run:
        for d in (train_input, train_target, val_input, val_target):
            d.mkdir(parents=True, exist_ok=True)

    def link_pairs(pair_list, inp_dir, tgt_dir, desc):
        for inp, gt in tqdm(pair_list, desc=desc, unit="img", dynamic_ncols=True):
            inp_link = inp_dir / inp.name
            gt_link = tgt_dir / inp.name
            if inp_link.is_symlink() or gt_link.is_symlink():
                stats["skipped_exists"] += 1
                continue
            make_relative_symlink(inp.resolve(), inp_link, dry_run)
            make_relative_symlink(gt.resolve(), gt_link, dry_run)

    link_pairs(train_pairs, train_input, train_target, "Symlinking train")
    link_pairs(val_pairs, val_input, val_target, "Symlinking val  ")

    # Summary
    total = len(input_files)
    matched_total = (
        stats["matched_unique"] + stats["ambiguous_confident"] + stats["ambiguous_uncertain"]
    )
    print("\n" + "=" * 55)
    print(f"  Total input files         : {total}")
    print(f"  Matched (unique key)      : {stats['matched_unique']}")
    print(f"  Ambiguous -> confident     : {stats['ambiguous_confident']}")
    print(f"  Ambiguous -> UNCERTAIN     : {stats['ambiguous_uncertain']}  <- manual review")
    print(f"  Unmatched (no key parsed) : {stats['unmatched_no_key']}")
    print(f"  Unmatched (key not in GT) : {stats['unmatched_not_found']}")
    print(f"  Skipped (link exists)     : {stats['skipped_exists']}")
    print(f"  Total paired              : {matched_total} / {total}")
    print(f"  Train / Val               : {len(train_pairs)} / {len(val_pairs)}")
    print("=" * 55)

    if unmatched_cases:
        print(f"\nUnmatched ({len(unmatched_cases)}):")
        for f in unmatched_cases[:30]:
            print(f"  {f}")
        if len(unmatched_cases) > 30:
            print(f"  ... and {len(unmatched_cases) - 30} more")
    if uncertain_cases:
        print(f"\nUncertain SYNTHIA matches ({len(uncertain_cases)}) — manual review required:")
        for c in uncertain_cases:
            print(f"  INPUT : {c['input']}")
            print(f"  CHOSEN: {c['chosen_gt']}")
            for cand in c["candidates"]:
                print(f"    cand: {cand}")
            print()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Create SYNDOF input/gt symlink pairs (train/val)."
    )
    parser.add_argument(
        "--gt-root", type=Path, required=True, help="Raw GT root (.../synthetic_datasets/image)."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Synthesized blurry inputs root (.../SYNDOF/image).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/SYNDOF"),
        help="Output root; builds {train,val}/{input,target} under it.",
    )
    parser.add_argument("--train-size", type=int, required=True, help="Pairs assigned to train.")
    parser.add_argument("--psnr-threshold", type=float, default=DEFAULT_PSNR_THRESHOLD)
    parser.add_argument("--dry-run", action="store_true", help="Preview without creating symlinks.")
    args = parser.parse_args()
    make_syndof(
        args.gt_root,
        args.input_root,
        args.out_root,
        args.train_size,
        dry_run=args.dry_run,
        psnr_threshold=args.psnr_threshold,
    )


if __name__ == "__main__":
    main()
