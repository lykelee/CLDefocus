"""
Batch test runner over the (model x weight x dataset) matrix.

Each CSV row is one combination: model, weight, dataset.
The runner composes a run from orthogonal pieces — no per-combo config files:
  - arch    : configs/arch/{model}.yaml          (model-only params)
  - dataset : configs/datasets/{dataset}.yaml    (img_path + gt_path)
  - weight  : a name looked up in configs/test/weights.yaml -> checkpoint path
              (a raw checkpoint path also works). Place checkpoints under weights/
              yourself — nothing is downloaded.
  - output  : images -> experiments/eval/img/{model}/{weight}/{dataset}
              metric -> experiments/eval/met/{model}/{weight}/{dataset}

The `weight` name doubles as the output-dir label (what used to be `tag`), so
one weight name -> one output folder. Dataset paths resolve under the dataset-view
root $DATA_ROOT (default `data`) — see resolve_dataset_path.

Phases:
  --infer    : test_{model}.py --config arch --img-path .. --gt-path .. \
                               --weights .. --output-dir img_dir   (images -> img_dir)
  --metric   : eval_files --pred img_dir --gt gt --out met_dir --metrics <file>
  --aggregate: gather each job's met_dir/results.csv -> one combined CSV at CSV
               (model,train,test,metric,value); no compute, just reads files.
Default (no --infer/--metric/--aggregate) -> infer + metric. --aggregate runs only
when its path is given. --skip-existing skips a phase whose output already exists.

CSV (header optional; '#'/blank lines ignored; column order = output nesting):
    model,weight,dataset
    nrknet,train_ours,realdof
    iniknet,train_dpdd,dpdd

Metrics: inline via --metrics psnr ssim lpips, or a text file via
    --metrics-file (one per line, '#' comments); --metrics wins if both given.

GPU is external (inherited by each child):
    CUDA_VISIBLE_DEVICES=0 uv run scripts/run_tests.py --list configs/test/full.csv
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_YAML = REPO_ROOT / "configs" / "test" / "weights.yaml"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_weights() -> dict:
    """model -> weight-name -> checkpoint path, from configs/test/weights.yaml."""
    if WEIGHTS_YAML.exists():
        return yaml.safe_load(WEIGHTS_YAML.read_text(encoding="utf-8")) or {}
    return {}


def resolve_weight(model: str, weight: str, weights: dict) -> str:
    """Map a CSV weight cell to a checkpoint path.

    A name in weights.yaml -> its path; a raw path (contains / or ends .pth) is
    used as-is; the sentinel '-' (nomodel baseline) passes through. Download
    nothing — place checkpoints under weights/ yourself."""
    if weight == "-":
        return weight
    entry = weights.get(model, {}).get(weight)
    if entry is not None:
        return str(entry)
    if "/" in weight or "\\" in weight or weight.endswith(".pth"):
        return weight
    raise KeyError(
        f"weight '{weight}' not found for model '{model}' in configs/test/weights.yaml "
        f"and is not a path. Add it there or pass a checkpoint path."
    )


def resolve_dataset_path(value: str) -> str:
    """Resolve a dataset path from configs/datasets/*.yaml. Values are written
    relative to the dataset-view root ($DATA_ROOT, default `data`), e.g.
    `DPDD/test/input`. Absolute values are used as-is. Relative $DATA_ROOT is
    resolved against the repo root (subprocess cwd)."""
    if Path(value).is_absolute():
        return value
    root = os.environ.get("DATA_ROOT", "").strip() or "data"
    return str(Path(root) / value)


def _has_images(dir_path: Path) -> bool:
    """True if dir_path holds at least one image file (used for --skip-existing)."""
    return dir_path.is_dir() and any(
        p.suffix.lower() in IMAGE_SUFFIXES for p in dir_path.iterdir() if p.is_file()
    )


def parse_rows(list_path: Path):
    """Yield (model, dataset, weight) from CSV; skip blanks/comments/header.

    Column order matches the output-dir nesting model/weight/dataset:
        model,weight,dataset
    `weight` is a registry name (configs/test/weights.yaml) or a raw checkpoint path."""
    with list_path.open(newline="", encoding="utf-8") as f:
        for raw in csv.reader(f):
            if not raw:
                continue
            cells = [c.strip() for c in raw]
            if not cells[0] or cells[0].startswith("#"):
                continue
            if len(cells) < 3 or not all(cells[:3]):
                print(f"  ! skip malformed row (need model,weight,dataset): {raw}", file=sys.stderr)
                continue
            model, weight, dataset = cells[:3]
            if model.lower() == "model" and weight.lower() == "weight":
                continue  # header
            yield model, dataset, weight


def parse_metrics(path: Path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def dataset_paths(dataset: str):
    """Return (img_path, gt_path) from configs/datasets/{dataset}.yaml, or None.

    `gt_path` is optional: a dataset without ground truth returns gt=None. Such a
    dataset supports no-reference metrics only — eval_files errors if a
    full-reference metric is requested without gt."""
    p = REPO_ROOT / "configs" / "datasets" / f"{dataset}.yaml"
    if not p.exists():
        return None
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    img, gt = d.get("img_path"), d.get("gt_path")
    if not img:
        return None
    gt = resolve_dataset_path(gt) if gt else None
    return resolve_dataset_path(img), gt


def run(cmd, dry):
    print(f"  $ {' '.join(cmd)}")
    if dry:
        return "dry"
    rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
    return "ok" if rc == 0 else f"FAIL({rc})"


def read_metric_means(results_csv: Path) -> dict:
    """Per-metric full-precision mean from a results.csv (filename + metric cols).
    results.csv is space-aligned -> read with skipinitialspace."""
    import math

    with results_csv.open(newline="", encoding="utf-8") as f:
        table = list(csv.reader(f, skipinitialspace=True))
    if len(table) < 2:
        return {}
    header, data = table[0], table[1:]
    means = {}
    for ci, name in enumerate(header):
        if ci == 0:  # filename column
            continue
        vals = []
        for r in data:
            if ci < len(r):
                try:
                    x = float(r[ci])
                except ValueError:
                    continue
                if math.isfinite(x):
                    vals.append(x)
        if vals:
            means[name] = sum(vals) / len(vals)  # full precision; str() round-trips losslessly
    return means


def write_aligned(header: list, rows: list, path: Path) -> None:
    """Space-aligned CSV (pandas read_csv skipinitialspace=True). Cells are
    written verbatim as strings -> values keep full precision (no truncation)."""
    table = [[str(v) for v in header]] + [[str(v) for v in r] for r in rows]
    ncol = len(table[0])
    widths = [max(len(row[c]) for row in table) for c in range(ncol)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        for row in table:
            parts = [
                (v + ",").ljust(widths[c] + 1) if c < ncol - 1 else v for c, v in enumerate(row)
            ]
            f.write("".join(parts).rstrip() + "\n")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--list", type=Path, help="CSV: model,weight,dataset rows.")
    ap.add_argument(
        "--row",
        action="append",
        default=[],
        metavar="model,weight,dataset",
        help="Inline combination (repeatable); use instead of / on top of --list. "
        "e.g., --row iniknet,train_ours,dpdd",
    )
    ap.add_argument("--infer", action="store_true", help="Run image inference.")
    ap.add_argument("--metric", action="store_true", help="Run metric evaluation.")
    ap.add_argument(
        "--aggregate",
        type=Path,
        metavar="CSV",
        help="Gather every job's results.csv into one combined CSV at this path "
        "(no compute). Omit to skip aggregation.",
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        metavar="NAME",
        help="Metric names inline (e.g., --metrics psnr ssim lpips); "
        "overrides --metrics-file when given.",
    )
    ap.add_argument(
        "--metrics-file",
        type=Path,
        default=REPO_ROOT / "configs" / "metrics" / "simple.txt",
        help="Text file listing metrics (one per line, '#' comments). "
        "Used when --metrics is not given.",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a phase for a job whose output already exists "
        "(images for infer, results.csv for metric). Default: recompute.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print commands, run nothing.")
    args = ap.parse_args()

    do_infer, do_metric = args.infer, args.metric
    do_aggregate = args.aggregate is not None
    if not (do_infer or do_metric or do_aggregate):
        do_infer = do_metric = True  # default: infer + metric (aggregate only with --aggregate)

    if not args.list and not args.row:
        print("give --list <csv> and/or --row model,weight,dataset", file=sys.stderr)
        sys.exit(1)
    rows = []
    if args.list:
        if not args.list.exists():
            print(f"list file not found: {args.list}", file=sys.stderr)
            sys.exit(1)
        rows += list(parse_rows(args.list))
    for spec in args.row:
        cells = [c.strip() for c in spec.split(",")]
        if len(cells) < 3 or not all(cells[:3]):
            print(f"  ! skip malformed --row (need model,weight,dataset): {spec}", file=sys.stderr)
            continue
        model, weight, dataset = cells[:3]
        rows.append((model, dataset, weight))
    if not rows:
        print("No valid rows.", file=sys.stderr)
        sys.exit(1)

    metrics = []
    if do_metric:
        if args.metrics:
            metrics = args.metrics
        else:
            if not args.metrics_file.exists():
                print(f"metrics file not found: {args.metrics_file}", file=sys.stderr)
                sys.exit(1)
            metrics = parse_metrics(args.metrics_file)
        if not metrics:
            src = "--metrics" if args.metrics else args.metrics_file
            print(f"no metrics ({src})", file=sys.stderr)
            sys.exit(1)

    weight_registry = load_weights()

    # Resolve every combination up-front (weight -> checkpoint path is deferred
    # to inference so metric-only / aggregate runs don't need the checkpoint).
    jobs = []  # dict per row
    for model, dataset, weight in rows:
        arch = REPO_ROOT / "configs" / "arch" / f"{model}.yaml"
        script = REPO_ROOT / "scripts" / f"test_{model}.py"
        ds = dataset_paths(dataset)
        img_dir = f"experiments/eval/img/{model}/{weight}/{dataset}"  # images -> img_dir/images
        met_dir = f"experiments/eval/met/{model}/{weight}/{dataset}"  # metric csv -> met_dir
        jobs.append(
            {
                "label": f"{model}/{weight}/{dataset}",
                "model": model,
                "weight": weight,
                "dataset": dataset,
                "script": script,
                "arch": arch,
                "ds": ds,
                "img_dir": img_dir,
                "met_dir": met_dir,
                "infer": "-",
                "metric": "-",
            }
        )

    # --- Phase 1: inference ---
    if do_infer:
        print("\n##### INFERENCE #####")
        for i, j in enumerate(jobs, 1):
            print(f"\n[infer {i}/{len(jobs)}] {j['label']}")
            if args.skip_existing and _has_images(REPO_ROOT / j["img_dir"]):
                print(f"  cached — images already in {j['img_dir']}")
                j["infer"] = "cached"
                continue
            missing = [str(p) for p in (j["script"], j["arch"]) if not p.exists()]
            if j["ds"] is None:
                missing.append(f"configs/datasets/*.yaml (or missing img_path key)")
            if missing:
                print(f"  ! skip — missing: {', '.join(missing)}", file=sys.stderr)
                j["infer"] = "skip"
                continue
            try:
                weight_path = resolve_weight(j["model"], j["weight"], weight_registry)
            except (KeyError, ValueError) as e:
                print(f"  ! skip — weight: {e}", file=sys.stderr)
                j["infer"] = "skip"
                continue
            img, gt = j["ds"]
            # Inference produces predictions only and ignores gt; reuse img as the
            # gt-path placeholder for gt-less datasets so the test script's dataset loads.
            cmd = [
                "uv",
                "run",
                "python",
                str(j["script"]),
                "--config",
                str(j["arch"]),
                "--img-path",
                img,
                "--gt-path",
                gt or img,
                "--weights",
                weight_path,
                "--output-dir",
                j["img_dir"],
            ]
            j["infer"] = run(cmd, args.dry_run)

    # --- Phase 2: metrics ---
    if do_metric:
        print("\n##### METRICS #####")
        for i, j in enumerate(jobs, 1):
            print(f"\n[metric {i}/{len(jobs)}] {j['label']}")
            if args.skip_existing and (REPO_ROOT / j["met_dir"] / "results.csv").exists():
                print(f"  cached — {j['met_dir']}/results.csv exists")
                j["metric"] = "cached"
                continue
            if j["ds"] is None:
                print(f"  ! skip — dataset config missing", file=sys.stderr)
                j["metric"] = "skip"
                continue
            _, gt = j["ds"]
            cmd = [
                "uv",
                "run",
                "python",
                "-m",
                "deblurring.eval_files",
                "--pred",
                j["img_dir"],
                "--out",
                j["met_dir"],
                "--metrics",
                *metrics,
            ]
            if gt:  # no gt -> eval_files runs NR-only and errors if a FR metric is requested
                cmd += ["--gt", gt]
            j["metric"] = run(cmd, args.dry_run)

    # --- Aggregate: gather every job's existing results.csv into one CSV ---
    if do_aggregate and not args.dry_run:
        agg_rows = []  # [model, train(tag), test(dataset), metric, value]
        for j in jobs:
            results_csv = REPO_ROOT / j["met_dir"] / "results.csv"
            if not results_csv.exists():
                print(f"  ! aggregate skip — no results.csv: {j['label']}", file=sys.stderr)
                continue
            for metric, mean in read_metric_means(results_csv).items():
                agg_rows.append([j["model"], j["weight"], j["dataset"], metric, mean])
        if agg_rows:
            write_aligned(["model", "train", "test", "metric", "value"], agg_rows, args.aggregate)
            print(f"\naggregate -> {args.aggregate}  ({len(agg_rows)} rows)")

    # --- Summary ---
    print("\n" + "=" * 64)
    print(f"  summary  (infer={do_infer} metric={do_metric} aggregate={do_aggregate})")
    print("=" * 64)
    print(f"  {'model/weight/dataset':<38} {'infer':<9} {'metric':<9}")
    n_fail = 0
    for j in jobs:
        n_fail += sum(1 for s in (j["infer"], j["metric"]) if str(s).startswith("FAIL"))
        print(f"  {j['label']:<38} {j['infer']:<9} {j['metric']:<9}")
    print("=" * 64)
    print(f"  rows {len(jobs)} | fail {n_fail}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
