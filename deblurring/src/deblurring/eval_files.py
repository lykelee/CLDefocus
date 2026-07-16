"""
Universal file-based image quality evaluator powered by pyiqa.
Supports any pyiqa metric — full-reference and no-reference.

Default metrics: psnr ssim lpips
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

IMAGE_SUFFIXES = frozenset({".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"})

DEFAULT_METRICS = ["psnr", "ssim", "lpips"]


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    """Returns (1, 3, H, W) float32 in [0, 1] on device."""
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    return t.to(device=device)


# ---------------------------------------------------------------------------
# File pairing
# ---------------------------------------------------------------------------


class Pair(NamedTuple):
    rel: Path
    pred: Path
    gt: Path | None


def collect_images(root: Path) -> dict[Path, Path]:
    """
    Map relative path WITHOUT extension -> file, so pred/gt pair by name
    regardless of differing extensions (e.g., foo.png vs foo.jpg).
    """
    return {
        p.relative_to(root).with_suffix(""): p
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    }


def pair_images(pred_dir: Path, gt_dir: Path | None, pair_by: str = "auto") -> list[Pair]:
    """
    Pair pred/gt files.

    pair_by:
      name  -> match by filename (extension ignored); error on any mismatch.
      order -> ignore names, pair by sorted order; requires equal counts.
      auto  -> try name; if names don't all match, fall back to order.
    """
    preds = collect_images(pred_dir)
    if not preds:
        raise ValueError(f"No images found in pred_dir: {pred_dir}")

    if gt_dir is None:
        return [Pair(rel, path, None) for rel, path in sorted(preds.items())]

    gts = collect_images(gt_dir)
    if not gts:
        raise ValueError(f"No images found in gt_dir: {gt_dir}")

    names_match = set(preds) == set(gts)

    if pair_by == "name" or (pair_by == "auto" and names_match):
        if not names_match:
            missing_gt = sorted(set(preds) - set(gts))
            missing_pred = sorted(set(gts) - set(preds))
            errors = []
            if missing_gt:
                errors.append(
                    f"{len(missing_gt)} pred(s) have no matching GT "
                    f"(e.g., {', '.join(p.as_posix() for p in missing_gt[:5])})"
                )
            if missing_pred:
                errors.append(
                    f"{len(missing_pred)} GT(s) have no matching pred "
                    f"(e.g., {', '.join(p.as_posix() for p in missing_pred[:5])})"
                )
            raise ValueError("Pairing failed (--pair-by name):\n  " + "\n  ".join(errors))
        return [Pair(rel, preds[rel], gts[rel]) for rel in sorted(preds)]

    # order pairing (explicit, or auto fallback)
    if len(preds) != len(gts):
        raise ValueError(f"Order pairing needs equal counts: {len(preds)} pred vs {len(gts)} gt.")
    if pair_by == "auto":
        print(
            f"Note: filenames differ — pairing {len(preds)} images by sorted order.",
            file=sys.stderr,
        )
    pred_keys, gt_keys = sorted(preds), sorted(gts)
    return [Pair(pk, preds[pk], gts[gk]) for pk, gk in zip(pred_keys, gt_keys)]


# ---------------------------------------------------------------------------
# Metrics via pyiqa
# ---------------------------------------------------------------------------


def build_metrics(names: list[str], device: torch.device) -> dict[str, object]:
    """Create pyiqa metric objects. Exits with a helpful message on unknown names."""
    import pyiqa

    available = set(pyiqa.list_models())
    unknown = [n for n in names if n not in available]
    if unknown:
        print(
            f"Unknown metric(s): {unknown}\n"
            f'Run `python -c "import pyiqa; print(pyiqa.list_models())"` to see all options.',
            file=sys.stderr,
        )
        sys.exit(1)

    metrics = {}
    for name in names:
        print(f"  loading metric: {name}", flush=True)
        metrics[name] = pyiqa.create_metric(name, device=device)
    return metrics


def metric_mode(metric_obj) -> str:
    """Returns 'FR' or 'NR'. Fallback to 'NR' if attribute missing."""
    return getattr(metric_obj, "metric_mode", "NR")


def needs_reference(metrics: dict[str, object]) -> bool:
    return any(metric_mode(m) == "FR" for m in metrics.values())


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate(
    pairs: list[Pair],
    metrics: dict[str, object],
    device: torch.device,
) -> list[dict]:
    rows: list[dict] = []

    with torch.inference_mode():
        for pair in tqdm(pairs, desc="evaluating", ncols=80):
            pred_t = load_image(pair.pred, device)
            gt_t = load_image(pair.gt, device) if pair.gt is not None else None

            row: dict = {"filename": pair.rel.as_posix()}

            for name, metric in metrics.items():
                mode = metric_mode(metric)
                try:
                    if mode == "FR":
                        if gt_t is None:
                            raise RuntimeError(
                                f"Metric '{name}' is full-reference but --gt was not provided."
                            )
                        if pred_t.shape != gt_t.shape:
                            raise ValueError(
                                f"Shape mismatch for {pair.rel}: "
                                f"{tuple(pred_t.shape)} vs {tuple(gt_t.shape)}"
                            )
                        row[name] = float(metric(pred_t, gt_t).item())
                    else:
                        row[name] = float(metric(pred_t).item())
                except Exception as exc:
                    print(f"\nWarning: {name} failed on {pair.rel}: {exc}", file=sys.stderr)
                    row[name] = float("nan")

            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _cell(v) -> str:
    if isinstance(v, float):
        return f"{v:.6f}"
    return "" if v is None else str(v)


def write_aligned_csv(header: list, rows: list[list], path: Path) -> None:
    """
    Comma-separated, but each field is padded with trailing spaces so columns
    line up. Readable as-is and parseable by pandas with skipinitialspace=True.

        col1,  metric,  0.394410
    """
    table = [[_cell(v) for v in header]] + [[_cell(v) for v in r] for r in rows]
    ncol = len(table[0])
    widths = [max(len(row[c]) for row in table) for c in range(ncol)]
    with path.open("w", newline="", encoding="utf-8") as f:
        for row in table:
            parts = [
                (val + ",").ljust(widths[c] + 1) if c < ncol - 1 else val
                for c, val in enumerate(row)
            ]
            f.write("".join(parts).rstrip() + "\n")


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    write_aligned_csv(fieldnames, [[r.get(k) for k in fieldnames] for r in rows], path)


def write_summary_csv(rows: list[dict], metrics: dict[str, object], path: Path) -> None:
    """Write per-metric aggregate (mean/std/direction/count) to a CSV."""
    lower_better = {n: getattr(m, "lower_better", None) for n, m in metrics.items()}
    out_rows = []
    for name in metrics.keys():
        vals = np.array([r[name] for r in rows if name in r], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        mean = finite.mean() if finite.size else float("nan")
        std = finite.std() if finite.size else float("nan")
        out_rows.append([name, float(mean), float(std), lower_better[name], int(finite.size)])
    write_aligned_csv(["metric", "mean", "std", "lower_better", "count"], out_rows, path)


def print_summary(rows: list[dict], metrics: dict[str, object]) -> None:
    metric_names = list(metrics.keys())
    width = max(len(n) for n in metric_names)
    lower_better = {n: getattr(m, "lower_better", None) for n, m in metrics.items()}

    print(f"\n{'─' * 52}")
    print(f"  {'metric':<{width}}   mean        std       dir")
    print(f"{'─' * 52}")
    for name in metric_names:
        vals = np.array([r[name] for r in rows if name in r], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        mean = finite.mean() if finite.size else float("nan")
        std = finite.std() if finite.size else float("nan")
        lb = lower_better[name]
        direction = "↓" if lb else ("↑" if lb is False else "?")
        print(f"  {name:<{width}}   {mean:>9.4f}   {std:>8.4f}   {direction}")
    print(f"{'─' * 52}")
    print(f"  images: {len(rows)}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(name)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate deblurring output images using pyiqa metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pred", type=Path, required=True, help="Predicted images directory.")
    p.add_argument(
        "--gt",
        type=Path,
        default=None,
        help="Ground-truth directory (required for full-reference metrics).",
    )
    p.add_argument("--out", type=Path, required=True, help="Output directory for results.csv.")
    p.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        metavar="METRIC",
        help="pyiqa metric names. Mix FR and NR freely.",
    )
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--pair-by",
        type=str,
        default="auto",
        choices=["auto", "name", "order"],
        help="Pred/GT pairing: name (filename), order (sorted index), "
        "auto (name, fall back to order on mismatch).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    pred_dir = args.pred.resolve()
    gt_dir = args.gt.resolve() if args.gt else None
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Metrics: {args.metrics}")

    metrics = build_metrics(args.metrics, device)

    if needs_reference(metrics) and gt_dir is None:
        fr_names = [n for n, m in metrics.items() if metric_mode(m) == "FR"]
        print(f"Error: metrics {fr_names} are full-reference — --gt is required.", file=sys.stderr)
        sys.exit(1)

    pairs = pair_images(pred_dir, gt_dir, args.pair_by)
    print(f"Found {len(pairs)} image(s).")

    rows = evaluate(pairs, metrics, device)
    csv_path = out_dir / "results.csv"
    summary_path = out_dir / "summary.csv"
    write_csv(rows, csv_path)
    write_summary_csv(rows, metrics, summary_path)
    print_summary(rows, metrics)
    print(f"Saved -> {csv_path}")
    print(f"Saved -> {summary_path}")


if __name__ == "__main__":
    main()
