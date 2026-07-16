"""
Stage 6 (s6_blur) — BlurNode

Applies layered DoF blur, ISP simulation (saturation, noise, RAW->RGB),
and saves per-patch blurry/sharp PNG pairs.

Reads from:  deps[0].store_dir/patches    (PatchStore  — Stage 1 (s1_patch_crop))
             deps[2].store_dir/layers      (LayerStore  — Stage 4 (s4_patch_gen))
             deps[3].store_dir/psfs        (PsfStore    — Stage 5 (s5_psf_gen))
Writes to:   store_dir/input              (blurry RGB PNGs)
             store_dir/target             (sharp  RGB PNGs)

ISP pipeline (normal path)
--------------------------
    lin_blurry  -> Photo.linear_to_raw -> noise (RAW domain) -> Photo.raw_to_lin
                -> Photo.lin_to_rgb -> blurry_rgb
    lin_sharp   -> Photo.lin_to_rgb -> sharp_rgb

Both outputs are saved as float32 RGB PNGs.

GPU multiprocessing
-------------------
_process_batch() groups items by photo and calls run_mp_per_gpu, dispatching
one photo (= a group of patches) per GPU call.  This amortises per-photo
setup (provider.open()) across all patches of the photo.

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
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from datasyn.pipeline import StageNode
from datasyn.synthesis.utils.stage_prof import StageProf

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

# TEMP per-worker profiler (env SYN_S6PROF=1). One instance per worker process.
_PROF = StageProf("s6prof", "SYN_S6PROF")

if TYPE_CHECKING:
    from typing import Callable, Tuple

    from numpy.typing import NDArray

    from datasyn.jaxutils.typing import BoolArray
    from datasyn.pipeline import ItemProgress
    from datasyn.synthesis.layered.scene import LayeredDepthScene
    from datasyn.synthesis.providers import PhotoProvider, PhotoProviderSpec
    from datasyn.synthesis.stores.layer_store import LayerStore
    from datasyn.synthesis.stores.patch_store import PatchStore
    from datasyn.synthesis.stores.psf_store import PsfStore
    from datasyn.utils.parallel.mp_per_gpu import Context


# ── Config ─────────────────────────────────────────────────────────────────────


@dataclass
class BlurConfig:
    seed: int
    ablate_isp: bool = False
    show_progress: bool = True


# ── Subprocess shared data ─────────────────────────────────────────────────────


class _MPData(NamedTuple):
    patch_store: "PatchStore"
    layer_store: "LayerStore"
    psf_store: "PsfStore"
    progress: "ItemProgress"
    provider: "PhotoProvider"
    out_sharp_dir: Path
    out_blurry_dir: Path


# ── Module-level init / worker (must be picklable for spawn) ───────────────────


def _init(
    ctx: "Context[_MPData]",
    patch_store_dir: str,
    layer_store_dir: str,
    psf_store_dir: str,
    progress_dir: str,
    provider_spec: "PhotoProviderSpec",
    out_sharp_dir: str,
    out_blurry_dir: str,
    log_dir: str,
) -> _MPData:
    """Called once per GPU subprocess before any worker calls."""
    from datasyn.synthesis.stages.s5_psf_gen import _setup_file_logging

    _setup_file_logging(log_dir, ctx.device)
    log.debug(f"_init start (GPU {ctx.device})")

    from datasyn.jaxutils.configs import easy_synthesis_setup, set_preallocation

    log.debug(f"GPU {ctx.device}: easy_synthesis_setup + set_preallocation")
    easy_synthesis_setup()
    set_preallocation(False)

    log.debug(f"GPU {ctx.device}: importing JAX / stores / provider")
    import jax.numpy as jnp  # noqa: F401 — triggers JAX init after CUDA device is set

    from datasyn.pipeline import ItemProgress
    from datasyn.synthesis.common import IOMode
    from datasyn.synthesis.stores.layer_store import LayerStore
    from datasyn.synthesis.stores.patch_store import PatchStore
    from datasyn.synthesis.stores.psf_store import PsfStore

    out_sharp = Path(out_sharp_dir)
    out_blurry = Path(out_blurry_dir)
    out_sharp.mkdir(parents=True, exist_ok=True)
    out_blurry.mkdir(parents=True, exist_ok=True)

    log.debug(f"GPU {ctx.device}: building provider")
    result = _MPData(
        patch_store=PatchStore(patch_store_dir, mode=IOMode.READ),
        layer_store=LayerStore(layer_store_dir, mode=IOMode.READ),
        psf_store=PsfStore(psf_store_dir, mode=IOMode.READ),
        progress=ItemProgress(progress_dir),
        provider=provider_spec.build(),
        out_sharp_dir=out_sharp,
        out_blurry_dir=out_blurry,
    )
    log.debug(f"GPU {ctx.device}: _init complete.")
    return result


def _worker(
    ctx: "Context[_MPData]",
    photo: str,
    photo_to_items: dict[str, list[str]],
    seed: int,
    ablate_isp: bool,
    fail_fast: bool,
) -> None:
    """Process all ready patches belonging to one photo."""
    import time

    import jax.numpy as jnp
    import jax.random as jr
    import numpy as np

    from datasyn.jaxutils import jx
    from datasyn.jaxutils.imgutils import arr1f_to_pil
    from datasyn.jaxutils.random import HRng, tag_seed
    from datasyn.synthesis.layered.blur import layered_dof_composite
    from datasyn.synthesis.layered.scene import layers_from_f2b
    from datasyn.utils.img.shift import align_to_centroid_dft_layout

    data = ctx.data
    _PROF.device = ctx.device
    item_ids = photo_to_items[photo]
    dtype_jax = jnp.float32

    hrng_patch = HRng.from_seed(tag_seed(seed, "patch"))

    log.debug(f"GPU {ctx.device}: photo '{photo}' — {len(item_ids)} patch(es)")

    # ── Load photo once (shared across all patches of this photo) ──────────────
    photo_obj = data.provider.open(photo)

    # ── Per-patch processing ───────────────────────────────────────────────────

    for item_id in item_ids:
        if data.progress.is_done(item_id):
            continue

        log.debug(f"GPU {ctx.device}: item '{item_id}' start")
        try:
            with _PROF.t("load"):
                lin_u16 = data.patch_store.load_linear(item_id)
                idx_map = data.layer_store.load_clst_idx(item_id)
                psfs_f2b = data.psf_store.load_psfs(item_id).astype(dtype_jax)

            hrng_p = hrng_patch[item_id]

            if ablate_isp:
                # Inject RGB to linear space
                # We just additionally need to disable final lin -> rgb after synthesis
                lin_1f = lin_u16 / 65535
                lin_1f = jnp.asarray(photo_obj.lin_to_rgb(np.asarray(lin_1f)))
                lin_u16 = jnp.clip(lin_1f * 65535, 0, 65535).astype(jnp.uint16)

            # Rebuild the inpainted image layers online from the linear patch +
            # the stored cluster index map (LayerStore is depth-only now). The
            # index map guarantees the same layer count/order Stage 5 (s5_psf_gen) used.
            with _PROF.t("rebuild"):
                pat_u16 = build_layered_scene_online(lin_u16, idx_map)

            hrng_p = hrng_patch[item_id]

            # u16 -> float layers
            c_1f = jnp.astype(pat_u16.layers.colors_f2b / 65535, dtype_jax)
            a_1f = jnp.astype(pat_u16.layers.alphas_f2b / 65535, dtype_jax)
            pat = pat_u16.set_layers(layers_from_f2b(c_1f, a_1f))
            lin_1f = jnp.astype(lin_u16 / 65535, dtype_jax)

            hrng_p = hrng_patch[item_id]

            # u16 -> float layers
            c_1f = jnp.astype(pat_u16.layers.colors_f2b / 65535, dtype_jax)
            a_1f = jnp.astype(pat_u16.layers.alphas_f2b / 65535, dtype_jax)
            pat = pat_u16.set_layers(layers_from_f2b(c_1f, a_1f))
            lin_1f = jnp.astype(lin_u16 / 65535, dtype_jax)

            # Align PSF centroids (CZT output is almost but not perfectly centred).
            psfs_f2b = jx.vmap(lambda x: align_to_centroid_dft_layout(x))(psfs_f2b)

            # ── Saturation augmentation ────────────────────────────────────────

            pat_cur = pat
            lin_sharp_cur = lin_1f

            if not ablate_isp:
                hrng_sat = hrng_p["saturation"]
                scale = jr.uniform(hrng_sat.key_of("scale"), minval=0.0, maxval=4.0)

                colors = pat_cur.layers.colors_f2b
                satmask = jnp.clip(
                    20 * (jnp.min(colors, axis=3, keepdims=True) - 0.95), 0, 1
                )
                pat_cur = pat_cur.set_layers(
                    layers_from_f2b(colors + scale * satmask, pat_cur.layers.alphas_f2b)
                )
                satmask_sharp = jnp.clip(
                    20 * (jnp.min(lin_sharp_cur, axis=2, keepdims=True) - 0.95), 0, 1
                )
                lin_sharp_cur = lin_sharp_cur + scale * satmask_sharp

            # ── Layered DoF blur ───────────────────────────────────────────────

            with _PROF.t("blur"):
                lin_blurry = layered_dof_composite(
                    pat_cur.colors_f2b, pat_cur.alphas_f2b, psfs_f2b, dtype=dtype_jax
                )
                if _PROF.enabled:
                    jx.block_until_ready_all()  # barrier for honest timing

            # ── ISP: linear -> RAW -> noise -> RGB ───────────────────────────────

            _t_isp = time.perf_counter()
            if ablate_isp:
                # No RAW round-trip / noise (that is the ablation). But the
                # blurry layers were composited in LINEAR space, so still apply
                # the display color transform — otherwise blurry stays in linear
                # light (darker / lower-contrast) while sharp is already sRGB
                # (lin_sharp_cur was converted above), giving a color-space
                # mismatched pair. sharp is already sRGB so leave it as is.
                sharp_rgb = lin_sharp_cur
                blurry_rgb = lin_blurry
            else:
                # lin -> raw
                blurry_raw = jnp.asarray(
                    photo_obj.linear_to_raw(np.asarray(lin_blurry))
                )

                # Shot noise (Poisson) + read noise (Gaussian) in RAW domain.
                hrng_ns = hrng_p["noise"]
                beta1 = jr.uniform(
                    hrng_ns.key_of("beta1"),
                    minval=0.5e-5,
                    maxval=1.5e-5,
                )
                beta2 = jr.uniform(
                    hrng_ns.key_of("beta2"),
                    minval=0.5e-5,
                    maxval=1.5e-5,
                )
                I_shot = jr.poisson(hrng_ns.key_of("shot"), blurry_raw / beta1) * beta1
                blurry_raw = I_shot + beta2 * jr.normal(hrng_ns.key_of("read"))

                # raw -> linear -> rgb

                blurry_lin = jnp.asarray(photo_obj.raw_to_lin(np.asarray(blurry_raw)))
                blurry_rgb = jnp.asarray(photo_obj.lin_to_rgb(np.asarray(blurry_lin)))
                sharp_rgb = jnp.asarray(photo_obj.lin_to_rgb(np.asarray(lin_sharp_cur)))

            # host conversions above already materialize JAX arrays (honest time)
            _PROF.add("isp", time.perf_counter() - _t_isp)

            # ── Save PNGs ──────────────────────────────────────────────────────

            from PIL.PngImagePlugin import PngInfo

            with _PROF.t("save"):
                meta = PngInfo()
                meta.add_text("photo", photo)
                arr1f_to_pil(sharp_rgb).save(
                    data.out_sharp_dir / f"{item_id}.png", pnginfo=meta
                )
                arr1f_to_pil(blurry_rgb).save(
                    data.out_blurry_dir / f"{item_id}.png", pnginfo=meta
                )

            data.progress.mark_done(item_id)
            _PROF.item_done()
            log.debug(f"GPU {ctx.device}: item '{item_id}' done")

        except Exception:
            tb = traceback.format_exc()
            log.error(f"GPU {ctx.device}: item '{item_id}' FAILED\n{tb}")
            data.progress.mark_failed(item_id, tb)
            if fail_fast:
                raise


# ── BlurNode ───────────────────────────────────────────────────────────────────


class BlurNode(StageNode):
    """
    GPU stage: applies layered DoF blur + ISP to produce blurry/sharp RGB PNG pairs.

    store_dir layout::

        store_dir/
            input/      <- blurry RGB PNGs  (PairedImageDataset 'input' dir)
            target/     <- sharp  RGB PNGs  (PairedImageDataset 'target' dir)
            logs/       <- per-GPU log files
            _progress/  <- ItemProgress

    deps: [PatchCropNode, CameraChooseNode, PatchGenerateNode, PsfGenNode]
    Camera store (deps[1]) is not read directly; it is listed so the pipeline
    waits for Stage 3 (s3_camera) before running Stage 6 (s6_blur).
    """

    def __init__(
        self,
        name: str,
        store_dir: Path,
        cfg: BlurConfig,
        deps: list[StageNode],
        provider_spec: "PhotoProviderSpec",
        n_workers: int = 1,  # unused (GPU parallelism via run_mp_per_gpu)
    ) -> None:
        super().__init__(
            name,
            Path(store_dir),
            deps=deps,
            n_workers=n_workers,
        )
        self._cfg = cfg
        self._provider_spec = provider_spec

        # Store paths only — no JAX imports here.
        self._patch_store_dir = deps[0].store_dir / "patches"
        self._layer_store_dir = deps[2].store_dir / "layers"
        self._psf_store_dir = deps[3].store_dir / "psfs"

        # item->photo map, built lazily on first _process_batch call.
        self._item_to_photo: dict[str, str] | None = None

    def process(self, item_id: str) -> None:  # noqa: ARG002
        raise NotImplementedError(
            "BlurNode dispatches via run_mp_per_gpu in _process_batch(); "
            "process() is not used."
        )

    def _process_batch(self, item_ids: list[str], fail_fast: bool) -> None:
        from datasyn.jaxutils.random import tag_seed
        from datasyn.synthesis.common import IOMode
        from datasyn.synthesis.stores.patch_store import PatchStore
        from datasyn.utils.parallel import MPPerGPUConfig, run_mp_per_gpu

        # Build item->photo map once (bulk SQLite read, then cached).
        if self._item_to_photo is None:
            self._item_to_photo = PatchStore(
                self._patch_store_dir, mode=IOMode.READ
            ).load_photo_map()

        # Group this batch by photo.
        photo_to_items: dict[str, list[str]] = defaultdict(list)  # type: ignore[assignment]
        for item_id in item_ids:
            photo_to_items[self._item_to_photo[item_id]].append(item_id)
        photo_to_items = dict(photo_to_items)

        seed = tag_seed(self._cfg.seed, "blur")

        cfg_mp = MPPerGPUConfig(
            worker_func=_worker,
            indices=sorted(photo_to_items.keys()),
            worker_args=(
                photo_to_items,
                seed,
                self._cfg.ablate_isp,
                fail_fast,
            ),
            fn_init_proc=_init,
            init_args=(
                str(self._patch_store_dir),
                str(self._layer_store_dir),
                str(self._psf_store_dir),
                str(self.store_dir / "_progress"),
                self._provider_spec,
                str(self.store_dir / "target"),
                str(self.store_dir / "input"),
                str(self.store_dir / "logs"),
            ),
            show_progress=self._cfg.show_progress,
        )
        run_mp_per_gpu(cfg_mp)


_INPAINT_RADIUS = 3


def _estimate_layered_scene(
    image: NDArray, lvmap: NDArray, inpainter: Callable[[NDArray, BoolArray], NDArray]
):
    """lvmap: Level map, order: 0 (foreground) -> n-1 (background)."""
    import numpy as np

    from datasyn.jaxutils import wrappings as myjax
    from datasyn.synthesis.layered.scene import LayeredDepthScene, layers_from_f2b

    assert image.ndim >= 2 and lvmap.ndim == 2
    assert image.shape[:2] == lvmap.shape[:2]

    image = np.asarray(image)
    lvmap = np.asarray(lvmap)

    levels = np.unique(lvmap)
    n_layers = levels.shape[0]

    canvas_0 = image
    occluded_0 = np.zeros(lvmap.shape, dtype=bool)

    def step(
        carry: Tuple[NDArray, NDArray], i: int
    ) -> Tuple[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray]]:
        canvas, occluded = carry

        lv_visible = lvmap == levels[i]
        lv_mask = occluded | lv_visible
        lv_color = lv_mask[..., None] * canvas

        from scipy.ndimage import binary_dilation

        structure = np.ones((3, 3), dtype=bool)
        mask_inpaint = binary_dilation(lv_visible, iterations=1, structure=structure)
        canvas_ = inpainter(canvas, mask_inpaint)

        carry = (canvas_, lv_mask)
        return carry, (lv_color, lv_mask)

    _, (colors, masks) = myjax.fake_scan(
        step, (canvas_0, occluded_0), xs=np.arange(n_layers)
    )

    scene = layers_from_f2b(colors, 65535 * masks)
    return LayeredDepthScene(scene, levels)


def _inpaint_u16(image: NDArray, mask: NDArray) -> NDArray:
    """
    Simply inpaint the masked region of a uint16 image, per channel.

    `mask` is a bool map; `cv2.inpaint` leaves non-masked pixels untouched.
    """
    import cv2
    import numpy as np

    assert image.dtype == np.uint16
    mask8 = (255 * mask).astype(np.uint8)
    inpainted = [
        cv2.inpaint(image[:, :, c], mask8, _INPAINT_RADIUS, cv2.INPAINT_TELEA)
        for c in range(3)
    ]
    return np.stack(inpainted, axis=-1)


def build_layered_scene_online(lin_u16: NDArray, idx_map: NDArray) -> LayeredDepthScene:
    """
    Rebuild the inpainted layered scene from a linear patch + cluster index map.
    """
    import numpy as np

    return _estimate_layered_scene(
        np.asarray(lin_u16), np.asarray(idx_map), inpainter=_inpaint_u16
    )
