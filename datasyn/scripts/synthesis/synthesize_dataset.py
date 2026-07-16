"""
Entry point for the synthesis pipeline. Please see `README.md`.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

STAGE_CHOICES = ("s1", "s2", "s3", "s4", "s5", "s6")


def _make_s2_sharpness_filter_node(
    name: str,
    store_dir: Path,
    deps: list[Any],
    cfg: Any,
):
    from datasyn.synthesis.stages.s2_sharpness_filter import SharpnessFilterNode

    return SharpnessFilterNode(name, store_dir, cfg, deps=deps)


def _make_s4_patch_generate_node(
    name: str,
    store_dir: Path,
    deps: list[Any],
    cfg: Any,
    n_workers: int,
):
    from datasyn.synthesis.stages.s4_patch_gen import PatchGenerateNode

    return PatchGenerateNode(name, store_dir, cfg, deps=deps, n_workers=n_workers)


def _path(value: Any, *, field: str) -> Path:
    if value is None:
        raise ValueError(f"Missing path config value: {field}")
    return Path(value).expanduser()


def _optional_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value).expanduser()


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Config section `{key}` must be a mapping.")
    return value


def _tuple_int2(value: Any, *, field: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"`{field}` must be a two-element list/tuple.")
    return int(value[0]), int(value[1])


def _tuple_float2(value: Any, *, field: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"`{field}` must be a two-element list/tuple.")
    return float(value[0]), float(value[1])


def _load_sharp_to_blurry(mapping_cfg: dict[str, Any] | None) -> dict[str, str] | None:
    if not mapping_cfg:
        return None

    target_list = _path(mapping_cfg.get("target_list"), field="target_list")
    source_list = _path(mapping_cfg.get("source_list"), field="source_list")
    with target_list.open("r", encoding="utf-8") as ft:
        sharps = sorted(ft.read().splitlines())
    with source_list.open("r", encoding="utf-8") as fs:
        blurrys = sorted(fs.read().splitlines())

    if len(sharps) != len(blurrys):
        raise ValueError(
            f"Sharp/blurry mapping list length mismatch: "
            f"{target_list} ({len(sharps)}) vs {source_list} ({len(blurrys)})"
        )
    return {s: b for s, b in zip(sharps, blurrys)}


def _provider_spec_from_config(
    provider_cfg: dict[str, Any],
    data_kind: str,
):
    from datasyn.synthesis.providers.various import NormalProviderSpec

    kind = provider_cfg.get("kind", "normal")

    # Params for the chosen kind live directly under `provider`. They may be
    # split per data_kind (train/val/test) or given flat; prefer the per-split
    # block when present.
    params = provider_cfg.get(data_kind)
    if not isinstance(params, dict):
        params = provider_cfg

    match kind:
        case "normal":
            return NormalProviderSpec(
                dir_raw=_optional_path(params.get("dir_raw")),
                dir_rgb=_optional_path(params.get("dir_rgb")),
                dir_dpt=_path(params.get("dir_dpt"), field="provider.dir_dpt"),
                dir_blurry_raw=_optional_path(params.get("dir_blurry_raw")),
                sharp_to_blurry=_load_sharp_to_blurry(
                    params.get("sharp_to_blurry_from")
                ),
            )
        case _:
            raise ValueError(f"Unknown provider kind: {kind!r}")


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    data_cfg = _section(config, "data")
    run_cfg = _section(config, "run")

    if args.type is not None:
        data_cfg["kind"] = args.type
    if args.stages is not None:
        run_cfg["stages"] = args.stages
    if args.workers is not None:
        run_cfg["workers"] = args.workers
    if args.fail_fast:
        run_cfg["fail_fast"] = True
    if args.show_progress is not None:
        run_cfg["show_progress"] = args.show_progress


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 0329 synthesis pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config path (required), e.g., configs/default.yaml.",
    )
    parser.add_argument("--type", choices=["train", "val", "test"], default=None)
    parser.add_argument(
        "--disable_progress_bars",
        "--no_pbars",
        dest="show_progress",
        action="store_false",
        default=None,
        help="Disable pipeline-level and inner GPU progress bars",
    )
    parser.add_argument(
        "--show_pbars",
        dest="show_progress",
        action="store_true",
        help="Enable pipeline-level and inner GPU progress bars",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker threads for S1 and CPU subprocess stages",
    )
    parser.add_argument(
        "--fail_fast",
        "--debug_exceptions",
        action="store_true",
        help="Re-raise item/stage exceptions instead of only recording failures.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_CHOICES,
        default=None,
        help="Which stages to run",
    )
    return parser.parse_args()


def _with_deps(node: Any, ordered: list[Any], seen: set[str]) -> None:
    if node.name in seen:
        return
    for dep in node.deps:
        _with_deps(dep, ordered, seen)
    seen.add(node.name)
    ordered.append(node)


def main():
    from datasyn.jaxutils.configs import set_preallocation
    from datasyn.synthesis.config import load_run_config

    set_preallocation(False)  # For subprocessing

    args = _parse_args()
    config = load_run_config(args.config)
    _apply_cli_overrides(config, args)

    data_cfg = _section(config, "data")
    run_cfg = _section(config, "run")
    lens_cfg = _section(config, "lens")
    split_cfgs = _section(config, "splits")

    data_kind = str(data_cfg.get("kind", "train"))
    dataset_name = str(data_cfg.get("dataset_name", "dpdd"))
    try:
        split_cfg = split_cfgs[data_kind]
    except KeyError as exc:
        raise KeyError(f"No split config for data.kind={data_kind!r}") from exc

    data_root = _path(data_cfg.get("root"), field="data.root")
    # Optional per-stage root overrides (for splitting heavy stages across disks).
    # Stages absent from `data.stage_roots` fall back to the common `data.root`.
    stage_roots = _section(data_cfg, "stage_roots")

    def _stage_root(stage_key: str) -> Path:
        override = stage_roots.get(stage_key)
        return (
            _path(override, field=f"data.stage_roots.{stage_key}")
            if override
            else data_root
        )

    workers = int(run_cfg.get("workers", 1))
    show_progress = bool(run_cfg.get("show_progress", True))

    dir_s1 = _stage_root("s1") / "s1_patch_crop"
    dir_s2 = _stage_root("s2") / "s2_sharpness_filter"
    dir_s3 = _stage_root("s3") / "s3_camera_choose"
    dir_s4 = _stage_root("s4") / "s4_patch_gen"
    dir_s5 = _stage_root("s5") / "s5_psf_gen"
    dir_s6 = _stage_root("s6") / "s6_blur"

    from datasyn.pipeline import Pipeline
    from datasyn.synthesis.stages.s1_patch_crop import PatchCropConfig
    from datasyn.synthesis.stages.s2_sharpness_filter import SharpnessFilterConfig
    from datasyn.synthesis.stages.s3_camera import CameraChooseConfig
    from datasyn.synthesis.stages.s4_patch_gen import PatchGenConfig
    from datasyn.synthesis.stages.s5_psf_gen import PsfGenConfig, PsfGenNode
    from datasyn.synthesis.stages.s6_blur import BlurConfig, BlurNode
    from datasyn.synthesis.subprocess_nodes import (
        SubprocessStageRunNode,
        cpu_only_jax_env,
    )

    provider_spec = _provider_spec_from_config(_section(config, "provider"), data_kind)

    s1_cfg_dict = _section(config, "s1")
    cfg_s1_kwargs: dict[str, Any] = {}
    if "aug_scale_range" in s1_cfg_dict:
        cfg_s1_kwargs["aug_scale_range"] = _tuple_float2(
            s1_cfg_dict["aug_scale_range"], field="s1.aug_scale_range"
        )
    if "aug_angle_range" in s1_cfg_dict:
        cfg_s1_kwargs["aug_angle_range"] = _tuple_float2(
            s1_cfg_dict["aug_angle_range"], field="s1.aug_angle_range"
        )
    for key in (
        "aug_mode",
        "aug_flip",
        "save_viz_image",
        "save_viz_depth",
    ):
        if key in s1_cfg_dict:
            cfg_s1_kwargs[key] = s1_cfg_dict[key]
    if "aug_exposure_range" in s1_cfg_dict:
        cfg_s1_kwargs["aug_exposure_range"] = _tuple_float2(
            s1_cfg_dict["aug_exposure_range"], field="s1.aug_exposure_range"
        )

    from datasyn.jaxutils.random import tag_seed

    # Single global seed. Stage streams are separated by per-stage tags
    # ("crop"/"camera"/"blur") inside each stage, not by separate seed fields.
    global_seed = int(run_cfg.get("seed", 0))

    # Per-patch dataset seed: folds the output-dataset labels (source name +
    # train/val/test kind). Orthogonal to the lens db on purpose, so swapping
    # the lens db keeps crop/focusing/noise draws identical.
    dataset_seed = tag_seed(tag_seed(global_seed, dataset_name), data_kind)

    # Lens-partition seed: tagged ONLY with the lens db identity (kind- and
    # dataset-independent), so train/val/test share one permutation and their
    # lens_choose ranges form a disjoint partition (no train/test lens leak).
    lens_db_name = str(lens_cfg.get("name") or lens_cfg.get("stats_db"))
    seed_lens = tag_seed(global_seed, lens_db_name)

    cfg_s1 = PatchCropConfig(
        seed=dataset_seed,
        patch_hw=_tuple_int2(
            s1_cfg_dict.get("patch_hw", [384, 384]), field="s1.patch_hw"
        ),
        n_patches_total=int(
            s1_cfg_dict.get("n_patches_total", split_cfg["n_patches_total"])
        ),
        n_candidates=int(s1_cfg_dict.get("n_candidates", split_cfg["n_candidates"])),
        n_unit_workers=workers,
        **cfg_s1_kwargs,
    )

    s2_cfg = _section(config, "s2")
    cfg_s2 = SharpnessFilterConfig(
        n_keep=int(s2_cfg.get("n_keep", split_cfg["n_keep"]))
    )
    assert cfg_s2.n_keep <= cfg_s1.n_patches_total

    s3_cfg = _section(config, "s3")
    category_filters = lens_cfg.get("category_filters")
    if category_filters is not None:
        category_filters = tuple(str(x) for x in category_filters)
    cfg_s3 = CameraChooseConfig(
        seed=dataset_seed,
        seed_lens=seed_lens,
        ks=int(s3_cfg.get("ks", 128)),
        imgsize_wh=_tuple_int2(
            s3_cfg.get("imgsize_wh", [1680, 1120]), field="s3.imgsize_wh"
        ),
        stats_db_path=_path(lens_cfg.get("stats_db"), field="lens.stats_db"),
        classify_db_path=_optional_path(lens_cfg.get("classify_db")),
        category_filters=category_filters,
        lens_choose_start=int(
            s3_cfg.get("lens_choose_start", split_cfg["lens_choose_start"])
        ),
        lens_choose_end=s3_cfg.get("lens_choose_end", split_cfg["lens_choose_end"]),
        gpu_task_chunk_size=int(s3_cfg.get("gpu_task_chunk_size", 32)),
        show_progress=show_progress,
    )

    s4_cfg = _section(config, "s4")
    cfg_s4 = PatchGenConfig(
        stats_db_path=_path(lens_cfg.get("stats_db"), field="lens.stats_db"),
        seed=dataset_seed,
        save_viz=bool(s4_cfg.get("save_viz", True)),
        save_coc=bool(s4_cfg.get("save_coc", True)),
        ablate_vardpt=bool(s4_cfg.get("ablate_vardpt", False)),
    )

    s5_cfg = _section(config, "s5")
    cfg_s5 = PsfGenConfig(
        ks=int(s5_cfg.get("ks", s3_cfg.get("ks", 128))),
        lens_root=_path(lens_cfg.get("root"), field="lens.root"),
        save_viz=bool(s5_cfg.get("save_viz", True)),
        ablate_psf=bool(s5_cfg.get("ablate_psf", False)),
        show_progress=show_progress,
    )

    s6_cfg = _section(config, "s6")
    cfg_s6 = BlurConfig(
        seed=dataset_seed,
        ablate_isp=bool(s6_cfg.get("ablate_isp", False)),
        show_progress=show_progress,
    )

    stages = [str(stage) for stage in run_cfg.get("stages", STAGE_CHOICES)]
    unknown_stages = sorted(set(stages) - set(STAGE_CHOICES))
    if unknown_stages:
        raise ValueError(f"Unknown stage key(s): {unknown_stages}")

    from datasyn.synthesis.stages.s1_patch_crop import PatchCropNode
    from datasyn.synthesis.stages.s3_camera import CameraChooseNode

    child_env_cpu_jax = cpu_only_jax_env()
    # s1 is GPU-visible and dispatches per-(photo, cycle) compute across one
    # process per GPU (gpu_pool=True). It is NOT wrapped in a CPU-only
    # subprocess — that would hide the GPUs from its worker pool. The
    # coordinator still commits in deterministic unit order.
    s1 = PatchCropNode("s1_patch_crop", dir_s1, cfg_s1, provider_spec, gpu_pool=True)
    s2 = SubprocessStageRunNode(
        "s2_sharpness_filter",
        dir_s2,
        deps=[s1],
        factory=_make_s2_sharpness_filter_node,
        factory_args=(cfg_s2,),
        child_env=child_env_cpu_jax,
        selected_ids_store_relpath="selected.db",
        selection_complete_relpath="selection_complete.json",
    )
    patch_stage = s1
    # S3 (camera) owns its GPU subprocess pool internally; do not wrap it in a
    # CPU-only SubprocessStageRunNode.
    s3 = CameraChooseNode(
        "s3_camera_choose",
        dir_s3,
        cfg_s3,
        deps=[patch_stage, s2],
        n_workers=1,
    )
    s4 = SubprocessStageRunNode(
        "s4_patch_gen",
        dir_s4,
        deps=[patch_stage, s3],
        factory=_make_s4_patch_generate_node,
        factory_args=(cfg_s4, workers),
        child_env=child_env_cpu_jax,
    )

    s5 = PsfGenNode(
        "s5_psf_gen",
        dir_s5,
        cfg_s5,
        deps=[patch_stage, s3, s4],
    )
    s6 = BlurNode(
        "s6_blur",
        dir_s6,
        cfg_s6,
        deps=[patch_stage, s3, s4, s5],
        provider_spec=provider_spec,
    )
    all_nodes = {
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "s4": s4,
        "s5": s5,
        "s6": s6,
    }

    requested_nodes = [all_nodes[k] for k in stages]
    nodes: list[Any] = []
    seen_node_names: set[str] = set()
    for node in requested_nodes:
        _with_deps(node, nodes, seen_node_names)

    pipe = Pipeline(nodes)
    print(f"Config: {args.config}")
    print(f"Pipeline: {pipe}")

    fail_fast = bool(run_cfg.get("fail_fast", False))
    print("Running batch...")
    pipe.run_batch(fail_fast=fail_fast, show_progress=show_progress)

    # Flush WAL files.
    for node in nodes:
        progress_dir = node.store_dir / "_progress"
        for db_path in sorted(progress_dir.glob("*.db")):
            with sqlite3.connect(db_path) as con:
                con.execute("PRAGMA wal_checkpoint(TRUNCATE);")

    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
