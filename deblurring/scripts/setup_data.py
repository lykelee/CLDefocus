"""
Build the repo-local data/ view from original downloaded datasets.

Reads data-paths.yaml (your real paths), then symlinks/builds each dataset into
data/<Name>/... exactly where the configs expect it. Run from the repo root.

    cp data-paths.example.yaml data-paths.yaml   # then edit paths
    uv run scripts/setup_data.py           # or: just setup-data
    uv run scripts/setup_data.py --only DPDD RTF --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from make_syndof_pairs import make_syndof  # noqa: E402

# Dataset-view root. Mirrors run_tests.resolve_dataset_path: $DATA_ROOT (default
# `data`); relative values resolve against the repo root, absolute used as-is.
_DATA_ROOT = os.environ.get("DATA_ROOT", "").strip() or "data"
DATA_DIR = Path(_DATA_ROOT) if Path(_DATA_ROOT).is_absolute() else REPO_ROOT / _DATA_ROOT
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

DRY = False
VERBOSE = False
_tally = {"created": 0, "skipped": 0}


def _images(d: Path):
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _symlink(target: Path, link: Path):
    """Create link -> target (absolute). Idempotent; skips existing links.
    Per-link output only with --verbose; otherwise counted into _tally."""
    target = target.resolve()
    if link.is_symlink() or link.exists():
        _tally["skipped"] += 1
        if VERBOSE:
            print(f"    skip (exists): {link.relative_to(REPO_ROOT)}")
        return
    _tally["created"] += 1
    if VERBOSE:
        print(f"    link {link.relative_to(REPO_ROOT)} -> {target}")
    if not DRY:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)


def _link_dir(src: Path, dst: Path):
    if not src.exists():
        raise FileNotFoundError(f"source not found: {src}")
    _symlink(src, dst)


def _link_paired_matched(src_dir: Path, tgt_dir: Path, out_input: Path, out_target: Path):
    """Pair by sorted order; name both links after the target file so input/target match."""
    srcs, tgts = _images(src_dir), _images(tgt_dir)
    if len(srcs) != len(tgts):
        raise ValueError(f"count mismatch {src_dir} ({len(srcs)}) vs {tgt_dir} ({len(tgts)})")
    if not DRY:
        out_input.mkdir(parents=True, exist_ok=True)
        out_target.mkdir(parents=True, exist_ok=True)
    for s, t in zip(srcs, tgts):
        _symlink(s, out_input / t.name)
        _symlink(t, out_target / t.name)
    print(
        f"    paired {len(srcs)} into {out_input.relative_to(REPO_ROOT)} / {out_target.relative_to(REPO_ROOT)}"
    )


# --- per-dataset handlers -------------------------------------------------


def setup_dpdd(root):
    root = Path(root)
    for split_raw, split in (("test_c", "test"), ("train_c", "train"), ("val_c", "val")):
        src, tgt = root / split_raw / "source", root / split_raw / "target"
        if not src.exists():
            print(f"    skip split {split_raw} (not found)")
            continue
        _link_paired_matched(
            src, tgt, DATA_DIR / "DPDD" / split / "input", DATA_DIR / "DPDD" / split / "target"
        )


def setup_realdof(root):
    _link_dir(Path(root), DATA_DIR / "RealDOF")


def setup_rtf(root):
    gtd = Path(root) / "Ground_Truth_Data"
    if not gtd.exists():
        raise FileNotFoundError(f"RTF: {gtd} not found")
    out_in, out_tg = DATA_DIR / "RTF" / "input", DATA_DIR / "RTF" / "target"
    for p in sorted(gtd.glob("image_*.png")):
        _symlink(p, out_in / p.name)
    for p in sorted(gtd.glob("sharp_*.png")):
        _symlink(p, out_tg / p.name)


def setup_cldefocus(root):
    _link_dir(Path(root), DATA_DIR / "CLDefocus")


def setup_s24(root):
    _link_dir(Path(root), DATA_DIR / "S24")


def setup_syndof(cfg):
    make_syndof(
        cfg["gt_root"], cfg["input_root"], DATA_DIR / "SYNDOF", int(cfg["train_size"]), dry_run=DRY
    )


HANDLERS = {
    "DPDD": setup_dpdd,
    "RealDOF": setup_realdof,
    "RTF": setup_rtf,
    "CLDefocus": setup_cldefocus,
    "S24": setup_s24,
    "SYNDOF": setup_syndof,
}


def main():
    global DRY, VERBOSE
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", type=Path, default=REPO_ROOT / "data-paths.yaml")
    ap.add_argument(
        "--only", nargs="+", metavar="NAME", help="Only these datasets (default: all present)."
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log every link (default: per-dataset summary).",
    )
    args = ap.parse_args()
    DRY = args.dry_run
    VERBOSE = args.verbose

    if not args.manifest.exists():
        print(
            f"manifest not found: {args.manifest}\n"
            f"cp data-paths.example.yaml data-paths.yaml and edit it.",
            file=sys.stderr,
        )
        sys.exit(1)
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8")) or {}

    wanted = args.only or list(manifest.keys())
    unknown = [n for n in wanted if n not in HANDLERS]
    if unknown:
        print(f"unknown dataset(s): {unknown}; known: {list(HANDLERS)}", file=sys.stderr)
        sys.exit(1)

    fails = 0
    verb = "would create" if DRY else "created"
    for name in wanted:
        if name not in manifest or manifest[name] in (None, ""):
            print(f"[{name}] not in manifest — skip")
            continue
        print(f"[{name}]")
        _tally["created"] = _tally["skipped"] = 0
        try:
            HANDLERS[name](manifest[name])
            print(f"    {_tally['created']} {verb}, {_tally['skipped']} already present")
        except Exception as e:
            print(f"    ERROR: {e}", file=sys.stderr)
            fails += 1
    print(f"\ndone ({len(wanted)} requested, {fails} failed)" + ("  [dry-run]" if DRY else ""))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
