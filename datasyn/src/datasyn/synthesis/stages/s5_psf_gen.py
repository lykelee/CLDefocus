"""
Stage 5 (s5_psf_gen) — PsfGenNode

Renders per-layer PSFs for each patch using the assigned lens.

Reads from:  deps[0].store_dir/patches    (PatchStore — Stage 1 (s1_patch_crop))
             deps[1].store_dir/cameras.db  (CameraStore — Stage 3 (s3_camera))
             deps[2].store_dir/layers      (LayerStore — Stage 4 (s4_patch_gen))
Writes to:   store_dir/psfs               (PsfStore)

Item ID: same patch name as Stages 1-3.

GPU multiprocessing
-------------------
_process_batch() groups items by photo and calls run_mp_per_gpu, dispatching
one photo (= a group of patches) per GPU call.  This amortises per-photo
setup (lens loading, paraxial property computation) across all patches that
share the same source photo.

_init() and _worker() are module-level functions so they can be pickled by
the spawn-based worker pool.

IMPORTANT: JAX must NOT be imported at the top level of this module.
           All JAX imports are deferred inside _init() and _worker().
"""

from __future__ import annotations

import logging
import traceback
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from datasyn.pipeline import StageNode
from datasyn.synthesis.utils.stage_prof import StageProf

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

# TEMP per-worker profiler (env SYN_S5PROF=1). One instance per worker process.
_PROF = StageProf("s5prof", "SYN_S5PROF")

# Per-PROCESS cache of (lens, sensor size) -> (imaging, jitted gen_wf). Module
# level (not inside _worker) so it persists across the many photos one worker
# subprocess handles; otherwise every photo would recompile its lenses. Each GPU
# worker is its own process, so each holds its own cache (no cross-talk).
_IMAGING_CACHE: dict = {}

if TYPE_CHECKING:
    from datasyn.optics.zernike import ZernikeConfig
    from datasyn.pipeline import ItemProgress
    from datasyn.synthesis.stores.camera_store import CameraStore
    from datasyn.synthesis.stores.layer_store import LayerStore
    from datasyn.synthesis.stores.patch_store import PatchStore
    from datasyn.synthesis.stores.psf_store import PsfStore
    from datasyn.utils.parallel.mp_per_gpu import Context


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class PsfGenConfig:
    ks: int  # PSF kernel size in pixels
    lens_root: Path  # root directory of lens .mytable files
    save_viz: bool = True  # save PSF grid PNGs for inspection
    show_progress: bool = True
    # Ablation: replace the real lens (Debye) PSF with a paraxial CoC Gaussian kernel.
    ablate_psf: bool = False


# ── Subprocess shared data ────────────────────────────────────────────────────


class _MPData(NamedTuple):
    patch_store: "PatchStore"
    camera_store: "CameraStore"
    layer_store: "LayerStore"
    psf_store: "PsfStore"
    progress: "ItemProgress"
    zer_cfg: "ZernikeConfig"
    viz_dir: Path | None


class _WorkerSummary(NamedTuple):
    photo: str
    done_item_ids: tuple[str, ...]
    failed_item_ids: tuple[str, ...]


# ── Module-level init / worker (must be picklable for spawn) ──────────────────


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


def _init(
    ctx: "Context[_MPData]",
    patch_store_dir: str,
    camera_store_path: str,
    layer_store_dir: str,
    psf_store_dir: str,
    progress_dir: str,
    viz_dir_str: str,
    log_dir: str,
) -> _MPData:
    """Called once per GPU subprocess before any worker calls."""

    _setup_file_logging(log_dir, ctx.device)
    log.debug(f"_init start (GPU {ctx.device})")

    from datasyn.jaxutils.configs import easy_optics_setup, set_preallocation

    log.debug(f"GPU {ctx.device}: easy_optics_setup + set_preallocation")
    easy_optics_setup()
    set_preallocation(True)

    log.debug(f"GPU {ctx.device}: importing JAX / optics")
    import datasyn.optics.zernike as zernlib
    from datasyn.pipeline import ItemProgress
    from datasyn.synthesis.common import IOMode
    from datasyn.synthesis.stores.camera_store import CameraStore
    from datasyn.synthesis.stores.layer_store import LayerStore
    from datasyn.synthesis.stores.patch_store import PatchStore
    from datasyn.synthesis.stores.psf_store import PsfStore

    log.debug(f"GPU {ctx.device}: building ZernikeConfig + JIT fn")
    zer_cfg = zernlib.make_zernike_config(n_max=15, kind=zernlib.ZernikeKind.REAL)
    viz_dir = Path(viz_dir_str) if viz_dir_str else None
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)

    log.debug(f"GPU {ctx.device}: opening stores")
    result = _MPData(
        patch_store=PatchStore(patch_store_dir, mode=IOMode.READ),
        camera_store=CameraStore(camera_store_path, mode=IOMode.READ),
        layer_store=LayerStore(layer_store_dir, mode=IOMode.READ),
        psf_store=PsfStore(psf_store_dir, mode=IOMode.WRITE),
        progress=ItemProgress(progress_dir),
        zer_cfg=zer_cfg,
        viz_dir=viz_dir,
    )
    log.debug(f"GPU {ctx.device}: _init complete.")
    return result


def _worker(
    ctx: "Context[_MPData]",
    photo: str,
    photo_to_items: dict[str, list[str]],
    lens_root: str,
    ks: int,
    ablate_psf: bool,
    fail_fast: bool,
) -> _WorkerSummary:
    """Process all ready patches belonging to one photo."""
    import jax.numpy as jnp

    from datasyn.jaxutils import jx
    from datasyn.jaxutils.imgutils import arr1f_to_pil, make_image_batch_2d_grid
    from datasyn.jaxutils.wrappings import fake_scan
    from datasyn.mathutils.dft.helpers import next_km_int, next_pow2_int
    from datasyn.optics.complens.tabular_lens import load_tabular_lens_from_mytable
    from datasyn.optics.imaging.analysis import gaussian_kernel
    from datasyn.optics.imaging.imaging import (
        DebyeProper,
        debye_sampling_bound_for_antialiasing_from_coords,
        make_imaging,
        make_tilted_debye_coords,
    )
    from datasyn.optics.ray import DEFAULT_WAVELENGTH
    from datasyn.synthesis.stores.psf_store import PSFMetadata
    from datasyn.utils.time import easy_timer

    disable_timer = True

    def jax_barrier():
        if not disable_timer:
            jx.block_until_ready_all()

    data = ctx.data
    _PROF.device = ctx.device
    item_ids = photo_to_items[photo]
    wvl = DEFAULT_WAVELENGTH
    done_item_ids: list[str] = []
    failed_item_ids: list[str] = []

    log.debug(f"GPU {ctx.device}: photo '{photo}' — {len(item_ids)} patch(es)")

    # Cache lens + imaging + the jitted wavefront generator per (lens, sensor
    # size) for the worker's lifetime. gen_wf was previously rebuilt every item,
    # forcing a fresh JIT compile of generate_wavefront per patch; caching it
    # reuses the compiled trace whenever a lens repeats (no output change).
    proper = DebyeProper(data.zer_cfg)

    def _build_gen_wf(imaging):
        @partial(jx.jit)
        def gen_wf(dpt, f_xy, wvl):
            return imaging.generate_wavefront(dpt, f_xy, wvl=wvl)

        return gen_wf

    for item_id in item_ids:
        if data.progress.is_done(item_id):
            continue  # already done (handles retry edge case)

        log.debug(f"GPU {ctx.device}: item '{item_id}' start")
        try:
            with _PROF.t("load"):
                patdef = data.patch_store.load_patdef(item_id)
                cam = data.camera_store.load(item_id)
                l2d = data.layer_store.load_depths(item_id)

            photo_hw = patdef.photo_size
            f_xy = jnp.asarray(patdef.f_xy)

            # ── Lens and paraxial setup (shared across layers of this patch) ──

            with _PROF.t("setup"):
                sen_wh = (photo_hw[1], photo_hw[0])
                cache_key = (cam.lens, sen_wh)
                entry = _IMAGING_CACHE.get(cache_key)
                if entry is None:
                    lens = load_tabular_lens_from_mytable(Path(lens_root) / cam.lens)
                    imaging = make_imaging(
                        lens, sen_wh=sen_wh, proper=proper, wvl_ref=wvl
                    )
                    entry = (imaging, _build_gen_wf(imaging))
                    _IMAGING_CACHE[cache_key] = entry
                    log.info(
                        "s5_psf_gen dev%s: compiled imaging for lens %s (cache=%d)",
                        ctx.device,
                        cam.lens,
                        len(_IMAGING_CACHE),
                    )
                imaging, gen_wf = entry

                s_prop = imaging.calc_sensor_s(cam.focusing, f_xy, wvl=wvl)

            # ── Render one PSF per depth layer ────────────────────────────────
            # gen_wf is compiled once per lens (cached above); ff is a thin,
            # un-jitted wrapper so it can take the per-layer dynamic MM.

            def ff(wf, s_prop: float, MM: int):
                return imaging.propagate_wavefront(wf, s_prop, ks=ks, MM=MM)

            def render_body(_, i_layer: int):
                dpt = l2d[i_layer]

                if ablate_psf:
                    # Ablation path: no lens simulation. Compute the CoC paraxially
                    # and render an isotropic Gaussian whose sigma matches that
                    # blur radius in pixels.
                    field = imaging.imghelp.dist_to_field(dpt)
                    rtm = imaging.parax.rtm
                    xp = imaging.parax.xp
                    s_ideal, _ = rtm.obj2img(rtm.z2e(field.z))
                    z_ideal = rtm.x2z(s_ideal)
                    z_prop = s_prop + xp.z
                    scoc = xp.r * (z_prop - z_ideal) / (z_ideal - xp.z)

                    pixsize = imaging.sen.grid.spacing  # (2,) pixel pitch
                    sigma = jnp.maximum(0.5 * jnp.abs(scoc) / pixsize, 0.25)
                    I_psf = gaussian_kernel(
                        ks,
                        mean=jnp.zeros((2,)),
                        cov=jnp.diag(sigma**2),
                        normalize=True,
                    )
                    return None, (I_psf, scoc)

                jax_barrier()
                with easy_timer(
                    "s5_psf_gen - generate_wavefront", disable=disable_timer
                ):
                    wf = gen_wf(dpt, f_xy, wvl=wvl)
                    jax_barrier()

                dcoords = make_tilted_debye_coords(
                    imaging.xp, wf.xp_sphere, z_obs=s_prop + imaging.xp.z
                )

                MM_min_pupil = 64
                MM_bin_unit = 256
                MM_limit = 2048
                MM_alias = (
                    2
                    * debye_sampling_bound_for_antialiasing_from_coords(
                        ior=1, wvl=wvl, coords=dcoords
                    )
                    + 1 * 256
                )

                MM = max(MM_alias, MM_min_pupil)
                MM = min(
                    next_pow2_int(MM), next_km_int(m=MM_bin_unit, n=MM)
                )  # To reduce JIT recompilation
                MM = int(min(MM, MM_limit))

                jax_barrier()
                with easy_timer(
                    f"s5_psf_gen - propagate_wavefront, MM = {MM}",
                    disable=disable_timer,
                ):
                    out = ff(wf, s_prop, MM=MM)
                    jax_barrier()

                I_psf = out.I_psf / out.I_psf.sum()

                return None, (I_psf, out.scoc)

            with _PROF.t("render"):
                _, (I_psf_arr, sgncoc_arr) = fake_scan(
                    render_body,
                    None,
                    jnp.arange(l2d.shape[0]),
                )  # shapes: (n_layers, ks, ks) and (n_layers,)
                if _PROF.enabled:
                    jx.block_until_ready_all()

            # ── Save ──────────────────────────────────────────────────────────

            with _PROF.t("store"):
                meta = PSFMetadata(lens=cam.lens, sgncoc=sgncoc_arr)
                data.psf_store.add_psfs(item_id, I_psf_arr, meta)

            if data.viz_dir is not None:
                with _PROF.t("viz"):
                    gridimg = make_image_batch_2d_grid(
                        I_psf_arr / jnp.max(I_psf_arr, axis=(1, 2), keepdims=True),
                        aspect=1.0,
                        border_width=2,
                        border_color=1.0,
                        pad_value=1.0,
                    )
                    arr1f_to_pil(gridimg).save(data.viz_dir / f"{item_id}.png")

            data.progress.mark_done(item_id)
            done_item_ids.append(item_id)
            _PROF.item_done()
            log.debug(f"GPU {ctx.device}: item '{item_id}' done")

        except Exception:
            tb = traceback.format_exc()
            log.error(f"GPU {ctx.device}: item '{item_id}' FAILED\n{tb}")
            data.progress.mark_failed(item_id, tb)
            failed_item_ids.append(item_id)
            if fail_fast:
                raise

    return _WorkerSummary(
        photo=photo,
        done_item_ids=tuple(done_item_ids),
        failed_item_ids=tuple(failed_item_ids),
    )


# ── PsfGenNode ─────────────────────────────────────────────────────────────────


class PsfGenNode(StageNode):
    """
    GPU stage: renders Zernike PSFs for every depth layer of every patch.

    store_dir layout::

        store_dir/
            psfs/      <- PsfStore (LMDB psfs + SQLite metadata)
            viz/       <- PSF grid PNGs per patch (when save_viz=True)
            _progress/ <- ItemProgress
    """

    def __init__(
        self,
        name: str,
        store_dir: Path,
        cfg: PsfGenConfig,
        deps: list[StageNode],
        n_workers: int = 1,  # unused (GPU parallelism via run_mp_per_gpu)
        exclusive_lock=None,
    ) -> None:
        super().__init__(
            name,
            Path(store_dir),
            deps=deps,
            n_workers=n_workers,
            exclusive_lock=exclusive_lock,
        )
        self._cfg = cfg

        # Store paths only — no JAX imports here.
        self._patch_store_dir = deps[0].store_dir / "patches"
        self._camera_store_path = deps[1].store_dir / "cameras.db"
        self._layer_store_dir = deps[2].store_dir / "layers"
        self._psf_store_dir = self.store_dir / "psfs"

        # item -> photo map, built lazily on first _process_batch call.
        self._item_to_photo: dict[str, str] | None = None

    def process(self, item_id: str) -> None:  # noqa: ARG002
        raise NotImplementedError(
            "PsfGenNode dispatches via run_mp_per_gpu in _process_batch(); "
            "process() is not used."
        )

    def _process_batch(self, item_ids: list[str], fail_fast: bool) -> None:
        from datasyn.synthesis.common import IOMode
        from datasyn.synthesis.stores.patch_store import PatchStore
        from datasyn.utils.parallel import MPPerGPUConfig, run_mp_per_gpu

        # Build item -> photo map once (bulk SQLite read, then cached).
        if self._item_to_photo is None:
            self._item_to_photo = PatchStore(
                self._patch_store_dir, mode=IOMode.READ
            ).load_photo_map()

        # Group this batch by photo.
        photo_to_items: dict[str, list[str]] = defaultdict(list)  # type: ignore[assignment]
        for item_id in item_ids:
            photo_to_items[self._item_to_photo[item_id]].append(item_id)
        photo_to_items = dict(photo_to_items)

        viz_dir_str = str(self.store_dir / "viz") if self._cfg.save_viz else ""

        cfg_mp = MPPerGPUConfig(
            worker_func=_worker,
            indices=sorted(photo_to_items.keys()),
            worker_args=(
                photo_to_items,
                str(self._cfg.lens_root),
                self._cfg.ks,
                self._cfg.ablate_psf,
                fail_fast,
            ),
            fn_init_proc=_init,
            init_args=(
                str(self._patch_store_dir),
                str(self._camera_store_path),
                str(self._layer_store_dir),
                str(self._psf_store_dir),
                str(self.store_dir / "_progress"),
                viz_dir_str,
                str(self.store_dir / "logs"),
            ),
            show_progress=self._cfg.show_progress,
        )
        summaries: list[_WorkerSummary] = run_mp_per_gpu(cfg_mp)

        failed_item_ids = sorted(
            item_id for summary in summaries for item_id in summary.failed_item_ids
        )
        if not failed_item_ids:
            return

        failed_photos = sorted(
            summary.photo for summary in summaries if summary.failed_item_ids
        )
        msg = (
            f"[{self.name}] GPU PSF generation recorded {len(failed_item_ids)} failed "
            f"item(s) across {len(failed_photos)} photo task(s): {failed_photos}. "
            "See errors.db and per-GPU logs for details."
        )
        if fail_fast:
            raise RuntimeError(msg)
        log.warning(msg)
