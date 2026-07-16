"""
Stage 3 (s3_camera) - CameraChooseNode

For each patch, samples a focusing distance and selects a compatible lens
based on CoC and Debye sampling criteria (paraxial optics).

Reads patch payloads from: deps[0].store_dir/patches  (PatchStore)
Selection/gating deps:     deps[1:]                  (ready-item gates only)
Writes to:                 store_dir/cameras.db      (CameraStore)

Item ID: same patch name as Stage 1 (s1_patch_crop).

GPU multiprocessing
-------------------
Camera/lens selection is dispatched through run_mp_per_gpu. JAX imports and
JAX-backed lens tensors are built only inside worker initialization so the
coordinator process does not allocate GPU memory.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd

from datasyn.pipeline import StageNode

if TYPE_CHECKING:
    from datasyn.jaxutils.random import HRng
    from datasyn.pipeline import ItemProgress
    from datasyn.synthesis.stores.camera_store import CameraStore
    from datasyn.synthesis.stores.patch_store import PatchStore
    from datasyn.utils.parallel.mp_per_gpu import Context

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)


@dataclass
class CameraChooseConfig:
    seed: int  # per-patch dataset seed (folds dataset_name + train/val/test kind)
    seed_lens: int  # lens-partition seed (folds lens db identity only)
    ks: int  # kernel size (determines max CoC limit)
    imgsize_wh: tuple[int, int]  # (width, height) of the full image
    stats_db_path: Path  # path to structured lens stats DB
    classify_db_path: Path | None = None  # optional structured classify DB
    category_filters: tuple[str, ...] | None = None  # optional union of categories
    wvl: float = field(default=0.0)  # 0.0 means DEFAULT_WAVELENGTH
    C_coc_limit_scale: float = 2.0
    C_debye_scale: float = 4.0
    N_debye_sam_limit: int = 1536
    foc_min: float = 0.2
    foc_max: float = 250.0
    lens_choose_start: int = 0
    lens_choose_end: int | None = None
    gpu_task_chunk_size: int = 32
    show_progress: bool = True


class _MPData(NamedTuple):
    patch_store: "PatchStore"
    camera_store: "CameraStore"
    progress: "ItemProgress"
    hrng_root: "HRng"
    lens_files: np.ndarray
    compute_values: object


class _WorkerSummary(NamedTuple):
    chunk_id: int
    done_item_ids: tuple[str, ...]
    failed_item_ids: tuple[str, ...]


def _setup_file_logging(log_dir: str, device: int) -> None:
    """Add a per-GPU FileHandler to the root logger (idempotent)."""
    log_path = Path(log_dir) / f"gpu_{device}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            f"%(asctime)s [GPU {device}] %(name)s %(levelname)s: %(message)s"
        )
    )
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)


def _load_lens_dataframe(cfg: CameraChooseConfig) -> pd.DataFrame:
    from datasyn.synthesis.lens.stats import STATS_COLUMNS
    from datasyn.utils.sqlite.tablestore import SQLiteTableStore

    stats_store = SQLiteTableStore(
        cfg.stats_db_path,
        STATS_COLUMNS,
        primary_key="file",
        readonly=True,
    )
    rows = stats_store.find(valid=True, crit=True)
    stats_store.close()
    df_all = pd.DataFrame(rows)

    if len(df_all) == 0:
        raise ValueError(f"No valid+crit lens rows found in '{cfg.stats_db_path}'")

    if cfg.category_filters is not None:
        if cfg.classify_db_path is None:
            raise ValueError(
                "CameraChooseConfig.category_filters requires classify_db_path."
            )
        from datasyn.synthesis.lens.classify import CLASSIFY_COLUMNS

        classify_store = SQLiteTableStore(
            cfg.classify_db_path,
            CLASSIFY_COLUMNS,
            primary_key="file",
            readonly=True,
        )
        rows_cls = classify_store.find()
        classify_store.close()
        df_cls = pd.DataFrame(rows_cls)
        allowed = df_cls[df_cls["category"].isin(tuple(cfg.category_filters))]["file"]
        df_all = df_all[df_all["file"].isin(allowed)]

    df_all = df_all.sort_values("file").reset_index(drop=True)

    if len(df_all) == 0:
        raise ValueError("Lens catalog is empty after s3_camera category filtering.")

    return df_all


def _build_lens_context(
    cfg: CameraChooseConfig,
    store_dir: str,
    write_lens_pool: bool,
):
    import jax.numpy as jnp
    import jax.random as jr

    from datasyn.jaxutils import jx
    from datasyn.jaxutils.random import HRng, tag_seed
    from datasyn.optics.paraxial import Pupil
    from datasyn.optics.paraxial.rtm2 import VirtualThinLens
    from datasyn.optics.ray import DEFAULT_WAVELENGTH
    from datasyn.synthesis.camera_choose import (
        find_debye_sampling_number,
        calc_signed_coc,
    )

    wvl = DEFAULT_WAVELENGTH if cfg.wvl == 0.0 else cfg.wvl

    # Per-patch stream (focusing, lens choice): dataset-labelled seed.
    hrng_root = HRng.from_seed(tag_seed(cfg.seed, "camera"))

    # Lens-pool permutation: lens-db-only seed (kind/dataset independent) so the
    # train/val/test lens_choose ranges slice one shared, disjoint partition.
    hrng_lens = HRng.from_seed(cfg.seed_lens)

    df_all = _load_lens_dataframe(cfg)
    shuffle = jr.permutation(hrng_lens.key_of("lens shuffle"), len(df_all))
    lens_idxs = shuffle[cfg.lens_choose_start : cfg.lens_choose_end]
    lens_idxs = np.asarray(jnp.sort(lens_idxs), dtype=np.int64)
    df = df_all.iloc[lens_idxs].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No lenses to use!")

    if write_lens_pool:
        lens_list_file = Path(store_dir) / "lens_pool.txt"
        lens_list_file.write_text("\n".join(df["name"]), encoding="utf-8")

    lens_files = np.asarray(df["file"])
    thinlens_bat = VirtualThinLens(
        power=jnp.asarray(df["power"]),
        fpp=jnp.asarray(df["fpp"]),
        bpp=jnp.asarray(df["bpp"]),
    )
    xp_bat = Pupil(z=jnp.asarray(df["xpz"]), r=jnp.asarray(df["xpr"]))

    w, h = cfg.imgsize_wh
    imgrad_bat = jnp.asarray(df["imgrad"])
    pixsize_bat = imgrad_bat * (2.0 / jnp.sqrt(w**2 + h**2))
    coc_limit_bat = pixsize_bat * (0.5 * cfg.ks / cfg.C_coc_limit_scale)
    tanhfov_bat = jnp.tan(jnp.deg2rad(jnp.asarray(df["hfov"])))

    fn_coc_vmap = jx.vmap(calc_signed_coc, in_axes=(0, 0, None, None))
    fn_debye_vmap = jx.jit(
        jx.vmap(
            lambda tl, xp, thfov, foc, dmin, dmax, fxy2: find_debye_sampling_number(
                tl,
                xp,
                thfov,
                foc,
                dmin,
                dmax,
                fxy2,
                wvl=wvl,
                C_debye_scale=cfg.C_debye_scale,
            ),
            in_axes=(0, 0, 0, None, None, None, None),
        )
    )

    n_debye = cfg.N_debye_sam_limit

    @jx.jit
    def compute_values(focusing, dpt_min, dpt_max, f_xy):
        sgncoc_1 = fn_coc_vmap(thinlens_bat, xp_bat, -dpt_min, -focusing)
        sgncoc_2 = fn_coc_vmap(thinlens_bat, xp_bat, -dpt_max, -focusing)
        sgncoc_max = jnp.where(
            jnp.abs(sgncoc_1) > jnp.abs(sgncoc_2), sgncoc_1, sgncoc_2
        )
        crit_coc = jnp.abs(sgncoc_max) <= coc_limit_bat
        f_xy2 = jnp.sum(jnp.asarray(f_xy) ** 2)
        sam = fn_debye_vmap(
            thinlens_bat,
            xp_bat,
            tanhfov_bat,
            focusing,
            dpt_min,
            dpt_max,
            f_xy2,
        )
        crit_debye = sam <= n_debye
        return sgncoc_max, crit_coc, crit_debye

    return hrng_root, lens_files, compute_values


def _init(
    ctx: "Context[_MPData]",
    patch_store_dir: str,
    camera_store_path: str,
    progress_dir: str,
    store_dir: str,
    log_dir: str,
    cfg: CameraChooseConfig,
) -> _MPData:
    """Called once per GPU subprocess before any worker calls."""
    _setup_file_logging(log_dir, ctx.device)
    log.info("GPU %s: initializing Stage 3 (s3_camera) camera chooser", ctx.device)

    from datasyn.jaxutils.configs import easy_optics_setup, set_preallocation

    easy_optics_setup()
    set_preallocation(val=False)

    from datasyn.pipeline import ItemProgress
    from datasyn.synthesis.common import IOMode
    from datasyn.synthesis.stores.camera_store import CameraStore
    from datasyn.synthesis.stores.patch_store import PatchStore

    hrng_root, lens_files, compute_values = _build_lens_context(
        cfg,
        store_dir=store_dir,
        write_lens_pool=ctx.id == 0,
    )

    return _MPData(
        patch_store=PatchStore(patch_store_dir, mode=IOMode.READ),
        camera_store=CameraStore(camera_store_path, mode=IOMode.WRITE),
        progress=ItemProgress(progress_dir),
        hrng_root=hrng_root,
        lens_files=lens_files,
        compute_values=compute_values,
    )


def _process_one(data: _MPData, cfg: CameraChooseConfig, item_id: str) -> None:
    import jax.numpy as jnp
    import jax.random as jr

    from datasyn.synthesis.camera_choose import safe_cluster
    from datasyn.synthesis.common import CameraParam
    from datasyn.synthesis.depth_sampling import (
        compute_positive_weights,
        sample_from_S_with_local_noise,
    )

    pat = data.patch_store.load_patdef(item_id)
    depth = data.patch_store.load_depth(item_id)

    # Keep the old sampling guard: clip invalid/out-of-range depth before
    # focus-distance sampling and lens feasibility checks.
    depth = np.clip(depth, cfg.foc_min, cfg.foc_max)

    hrng_pat = data.hrng_root[item_id]
    dpt_min = float(np.min(depth))
    dpt_max = float(np.max(depth))

    rng_foc = hrng_pat.key_of("focusing")
    A = 1.0 / (cfg.foc_max - cfg.foc_min + 1)
    B = 1.0
    depth_sorted = jnp.asarray(np.unique(depth), dtype=jnp.float32)
    inv_dpts = 1.0 / (depth_sorted - cfg.foc_min + 1)
    clst = safe_cluster(inv_dpts, eps=1e-2)
    S = jnp.unique(
        jnp.concat(
            [
                jnp.asarray([A]),
                jnp.clip(clst.clusters, A, B),
                jnp.asarray([B]),
            ]
        ),
        sorted=True,
    )
    weights = compute_positive_weights(S, A, B, max_interior_mass=0.4)
    inv_foc = sample_from_S_with_local_noise(
        rng_foc, S, weights, A, B, alpha=0.5, num_samples=1
    )[0]
    focusing = float((1.0 / inv_foc) - 1.0 + cfg.foc_min)

    sgncoc_max_bat, crit_coc, crit_debye = data.compute_values(
        focusing=focusing,
        dpt_min=dpt_min,
        dpt_max=dpt_max,
        f_xy=jnp.asarray(pat.f_xy),
    )
    crit_total = crit_coc & crit_debye

    if int(jnp.sum(crit_total)) == 0:
        raise ValueError(f"No valid lens for patch '{item_id}'")

    lens_idx = int(jr.choice(hrng_pat.key_of("lens"), jnp.nonzero(crit_total)[0]))
    lens = str(data.lens_files[lens_idx])
    sgncoc_max = float(sgncoc_max_bat[lens_idx])

    data.camera_store.add(
        item_id, CameraParam(lens=lens, focusing=focusing, sgncoc_max=sgncoc_max)
    )


def _worker(
    ctx: "Context[_MPData]",
    chunk: tuple[int, tuple[str, ...]],
    cfg: CameraChooseConfig,
    fail_fast: bool,
) -> _WorkerSummary:
    chunk_id, item_ids = chunk
    data = ctx.data
    done_item_ids: list[str] = []
    failed_item_ids: list[str] = []

    for item_id in item_ids:
        if data.progress.is_done(item_id):
            continue

        try:
            _process_one(data, cfg, item_id)
            data.progress.mark_done(item_id)
            done_item_ids.append(item_id)
        except Exception:
            tb = traceback.format_exc()
            log.error("GPU %s: item %r FAILED\n%s", ctx.device, item_id, tb)
            data.progress.mark_failed(item_id, tb)
            failed_item_ids.append(item_id)
            if fail_fast:
                raise

    return _WorkerSummary(
        chunk_id=chunk_id,
        done_item_ids=tuple(done_item_ids),
        failed_item_ids=tuple(failed_item_ids),
    )


def _make_chunks(
    item_ids: list[str], chunk_size: int
) -> list[tuple[int, tuple[str, ...]]]:
    chunk_size = max(1, int(chunk_size))
    ordered = sorted(item_ids)
    return [
        (i // chunk_size, tuple(ordered[i : i + chunk_size]))
        for i in range(0, len(ordered), chunk_size)
    ]


class CameraChooseNode(StageNode):
    """
    Downstream stage: assigns a lens and focusing distance to every patch.

    Dependency contract:
    - `deps[0]` must be the raw patch-producing stage so this node can read
      `PatchStore`.
    - Any later deps are treated as item-selection gates only. In the current
      filtered path, pass `s2_sharpness_filter` as `deps[1]`.

    store_dir layout::

        store_dir/
            cameras.db  -> CameraStore
            lens_pool.txt
            logs/
            _progress/  -> ItemProgress
    """

    def __init__(
        self,
        name: str,
        store_dir: Path,
        cfg: CameraChooseConfig,
        deps: list[StageNode],
        n_workers: int = 1,
    ) -> None:
        if len(deps) < 1:
            raise ValueError("CameraChooseNode requires deps[0] = patch stage")

        super().__init__(name, Path(store_dir), deps=deps, n_workers=n_workers)
        self._cfg = cfg
        self._patch_store_dir = deps[0].store_dir / "patches"
        self._camera_store_path = self.store_dir / "cameras.db"

        # TODO 260508: Quick-and-dirty to avoid "store-is-not-generated yet" for init of later stages!
        #              This should be revised in structure, not this quick-and-dirty.
        from datasyn.synthesis.common import IOMode
        from datasyn.synthesis.stores.camera_store import CameraStore

        CameraStore(self._camera_store_path, mode=IOMode.WRITE)

    def process(self, item_id: str) -> None:  # noqa: ARG002
        raise NotImplementedError(
            "CameraChooseNode dispatches via run_mp_per_gpu in _process_batch(); "
            "process() is not used."
        )

    def _process_batch(self, item_ids: list[str], fail_fast: bool) -> None:
        from datasyn.utils.parallel import MPPerGPUConfig, run_mp_per_gpu

        chunks = _make_chunks(item_ids, self._cfg.gpu_task_chunk_size)
        if not chunks:
            return

        cfg_mp = MPPerGPUConfig(
            worker_func=_worker,
            indices=chunks,
            worker_args=(self._cfg, fail_fast),
            fn_init_proc=_init,
            init_args=(
                str(self._patch_store_dir),
                str(self._camera_store_path),
                str(self.store_dir / "_progress"),
                str(self.store_dir),
                str(self.store_dir / "logs"),
                self._cfg,
            ),
            force_mp=True,
            show_progress=self._cfg.show_progress,
        )
        summaries: list[_WorkerSummary] = run_mp_per_gpu(cfg_mp)

        failed_item_ids = sorted(
            item_id for summary in summaries for item_id in summary.failed_item_ids
        )
        if not failed_item_ids:
            return

        failed_chunks = sorted(
            summary.chunk_id for summary in summaries if summary.failed_item_ids
        )
        msg = (
            f"[{self.name}] GPU camera selection recorded {len(failed_item_ids)} "
            f"failed item(s) across chunk(s) {failed_chunks}. "
            "See errors.db and per-GPU logs for details."
        )
        if fail_fast:
            raise RuntimeError(msg)
        log.warning(msg)
