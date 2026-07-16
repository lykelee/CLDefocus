"""
Stage 1 (s1_patch_crop) - PatchCropNode

Crops linear and depth patches using in-bounds rejection sampling and a
cycling strategy that targets an exact total patch count N.

Item IDs: "c{cycle:03d}__{photo}__{i:04d}"
  cycle -> which pass over the dataset this patch came from (0-based)
  photo -> source photo name
  i     -> acceptance index within this (photo, cycle) call (0-based)

Cycling strategy
----------------
work_units() = photo list from the provider (fixed order).
The node cycles over photos indefinitely until target_count() patches are
stored. Larger images admit more in-bounds candidate centers per cycle, so the
final distribution is naturally proportional to image area.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np

from datasyn.jaxutils.random import HRng, tag_seed
from datasyn.jaxutils.typing import RngKey
from datasyn.pipeline import ParallelWhileSourceNode
from datasyn.synthesis.common import IOMode
from datasyn.synthesis.providers import PhotoProviderSpec
from datasyn.synthesis.stages.s2_sharpness_filter import (
    compute_sharpness_raw_scores_batch,
)
from datasyn.synthesis.stores.patch_store import PatchDefine, PatchStore
from datasyn.synthesis.stores.sharpness_store import SharpnessStore
from datasyn.utils.files import ensure_directory


@dataclass(frozen=True)
class PatchAug:
    """Geometric and photometric augmentation parameters for a patch."""

    theta: float = 0.0
    scale: float = 1.0
    flip: int = 1
    exposure: float = 1.0


def aug_to_mat(aug: PatchAug):
    """Convert PatchAug to a 2x2 forward linear map (row, col space)."""
    from datasyn.jaxutils.utils import rot2d, scale2d

    return rot2d(aug.theta) @ scale2d(aug.scale, aug.flip * aug.scale)


def _random_circle_interior(key: RngKey) -> tuple[float, float]:
    """Sample (x, y) uniformly from the unit disk interior using JAX RNG."""
    import jax.numpy as jnp
    import jax.random as jr

    k_r, k_t = jr.split(key)
    r = jnp.sqrt(jr.uniform(k_r))
    theta = jr.uniform(k_t, minval=0.0, maxval=float(2.0 * math.pi))
    return float(r * jnp.cos(theta)), float(r * jnp.sin(theta))


def _make_item_id(cycle: int, photo: str, i: int) -> str:
    return f"c{cycle:03d}__{photo}__{i:04d}"


@dataclass
class PatchCropConfig:
    seed: int
    patch_hw: tuple[int, int]
    n_patches_total: int
    n_candidates: int
    aug_scale_range: tuple[float, float] = (0.7, 1.4)
    aug_angle_range: tuple[float, float] = (-math.pi, math.pi)
    aug_mode: Literal["full", "rigid8"] = "full"
    aug_flip: bool = True
    aug_exposure_range: tuple[float, float] = (0.7, 1.4)
    save_viz_image: bool = True
    save_viz_depth: bool = True
    n_unit_workers: int = 1


@dataclass
class _PatchCropItem:
    patdef: PatchDefine
    patch_lin_u16: np.ndarray
    patch_dpt: np.ndarray
    sharpness_raw: np.ndarray
    viz_rgb_u8: np.ndarray | None = None


@dataclass
class _PatchCropUnitResult:
    items: list[_PatchCropItem]


def _compute_patch_crop(
    provider,
    cfg: PatchCropConfig,
    hrng_root: HRng,
    photo: str,
    cycle: int,
) -> _PatchCropUnitResult:
    """
    Pure per-(photo, cycle) patch computation.

    Deterministic in (photo, cycle, item_id): all randomness derives from
    `hrng_root` keyed by semantic identifiers only — never worker/device.
    Used by both the coordinator (thread/serial fallback) and per-GPU workers.
    """
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from datasyn.jaxutils.utils import rot2d, scale2d
    from datasyn.synthesis.utils.sample_reject import (
        extract_patches,
        prepare_coords,
        preprocess_mask,
        sample_valid_centers,
    )

    photo_obj = provider.open(photo)
    depth = photo_obj.get_depth()
    H, W = int(depth.shape[0]), int(depth.shape[1])
    photo_hw: tuple[int, int] = (H, W)

    mask_data = preprocess_mask(jnp.ones((H, W), dtype=bool))

    hrng_unit = hrng_root["unit", photo, cycle]
    rigid8_only = cfg.aug_mode == "rigid8"

    N = cfg.n_candidates
    if rigid8_only:
        thetas = jr.choice(
            hrng_unit.key_of("theta"), (0.5 * jnp.pi) * jnp.arange(4), (N,)
        )
        scales = jnp.ones((N,), dtype=float)
    else:
        thetas = jr.uniform(
            hrng_unit.key_of("theta"),
            (N,),
            minval=float(cfg.aug_angle_range[0]),
            maxval=float(cfg.aug_angle_range[1]),
        )
        log_scales = jr.uniform(
            hrng_unit.key_of("scale"),
            (N,),
            minval=float(jnp.log(cfg.aug_scale_range[0])),
            maxval=float(jnp.log(cfg.aug_scale_range[1])),
        )
        scales = jnp.exp(log_scales)

    if cfg.aug_flip:
        flips = jnp.where(jr.uniform(hrng_unit.key_of("flip"), (N,)) < 0.5, -1.0, 1.0)
    else:
        flips = jnp.ones((N,), dtype=jnp.float32)

    if rigid8_only:
        exposures = jnp.ones((N,), dtype=jnp.float32)
    else:
        log_exposures = jr.uniform(
            hrng_unit.key_of("exposure"),
            (N,),
            minval=float(jnp.log(cfg.aug_exposure_range[0])),
            maxval=float(jnp.log(cfg.aug_exposure_range[1])),
        )
        exposures = jnp.exp(log_exposures)

    def _make_L_inv(theta, scale, flip):
        L_fwd = rot2d(theta) @ scale2d(scale, flip * scale)
        return jnp.linalg.inv(L_fwd)

    L_invs = jax.vmap(_make_L_inv)(thetas, scales, flips)

    sampled = sample_valid_centers(
        mask_data, cfg.patch_hw, L_invs, hrng_unit.key_of("center")
    )

    M = int(sampled.centers.shape[0])
    if M == 0:
        return _PatchCropUnitResult(items=[])

    img_lin = jnp.asarray(photo_obj.get_linear())

    prepared = prepare_coords(cfg.patch_hw, sampled.centers, sampled.L_invs)
    patches_lin = extract_patches(img_lin, prepared)
    patches_dpt = extract_patches(jnp.asarray(depth), prepared)

    valid_idx = np.where(np.array(sampled.valid_mask))[0]
    thetas_np = np.array(thetas)
    scales_np = np.array(scales)
    flips_np = np.array(flips).astype(int)
    exposures_np = np.array(exposures)

    valid_items: list[
        tuple[int, str, PatchAug, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    for i in range(M):
        item_id = _make_item_id(cycle, photo, i)
        orig_idx = int(valid_idx[i])
        aug = PatchAug(
            theta=float(thetas_np[orig_idx]),
            scale=float(scales_np[orig_idx]),
            flip=int(flips_np[orig_idx]),
            exposure=float(exposures_np[orig_idx]),
        )
        patch_dpt_np = np.asarray(patches_dpt[i], dtype=np.float32)
        if float(np.min(patch_dpt_np)) <= 0.0:
            continue

        patch_lin_np = np.clip(aug.exposure * np.asarray(patches_lin[i]), 0.0, 1.0)
        patch_rgb_np = np.asarray(photo_obj.lin_to_rgb(patch_lin_np), dtype=np.float32)
        valid_items.append((i, item_id, aug, patch_lin_np, patch_rgb_np, patch_dpt_np))

    if not valid_items:
        return _PatchCropUnitResult(items=[])

    sharpness_raw = compute_sharpness_raw_scores_batch(
        [item[4] for item in valid_items]
    )

    out_items: list[_PatchCropItem] = []
    for sharp_idx, (
        i,
        item_id,
        aug,
        patch_lin_np,
        patch_rgb_np,
        patch_dpt_np,
    ) in enumerate(valid_items):
        patch_lin_u16 = np.clip(65535.0 * patch_lin_np, 0, 65535).astype(np.uint16)

        hrng_item = hrng_root["item", item_id]
        fx, fy = _random_circle_interior(hrng_item.key_of("f_xy"))

        field_max = 0.7

        fx *= field_max
        fy *= field_max

        center = np.array(sampled.centers[i])
        patdef = PatchDefine(
            name=item_id,
            photo=photo,
            photo_size=photo_hw,
            patch_offset=(float(center[0]), float(center[1])),
            patch_size=cfg.patch_hw,
            f_xy=(fx, fy),
            aug=aug,
            depth_min=float(np.min(patch_dpt_np)),
            depth_max=float(np.max(patch_dpt_np)),
        )
        viz_rgb_u8 = None
        if cfg.save_viz_image:
            viz_rgb_u8 = np.clip(patch_rgb_np * 255.0, 0, 255).astype(np.uint8)

        out_items.append(
            _PatchCropItem(
                patdef=patdef,
                patch_lin_u16=patch_lin_u16,
                patch_dpt=patch_dpt_np,
                sharpness_raw=sharpness_raw[sharp_idx],
                viz_rgb_u8=viz_rgb_u8,
            )
        )

    return _PatchCropUnitResult(items=out_items)


class _S1WorkerData(NamedTuple):
    provider: object
    cfg: PatchCropConfig
    hrng_root: HRng


def _s1_gpu_init(ctx, provider_spec, cfg: PatchCropConfig) -> _S1WorkerData:
    """Per-GPU worker init: build provider + RNG once (JAX/GPU init here)."""
    from datasyn.jaxutils.configs import easy_optics_setup

    easy_optics_setup()
    return _S1WorkerData(
        provider=provider_spec.build(),
        cfg=cfg,
        hrng_root=HRng.from_seed(tag_seed(cfg.seed, "crop")),
    )


def _s1_gpu_worker(ctx, task: tuple[int, int, str]) -> _PatchCropUnitResult:
    """Per-GPU worker: compute one (photo, cycle). task = (cycle, unit_idx, photo)."""
    _cycle, _unit_idx, photo = task
    d: _S1WorkerData = ctx.data
    return _compute_patch_crop(d.provider, d.cfg, d.hrng_root, photo, _cycle)


class PatchCropNode(ParallelWhileSourceNode):
    """
    Source stage: crops candidate patches using in-bounds rejection sampling,
    cycling over photos until exactly n_patches_total patches are stored.

    store_dir layout::

        store_dir/
            patches/                 -> PatchStore
            sharpness.db             -> SharpnessStore
            viz/images/              -> display-RGB JPEG patches
            _progress/
                done.db              -> patch IDs produced so far
                cycle_checkpoint.db  -> (cycle, unit_idx) resume point
    """

    def __init__(
        self,
        name: str,
        store_dir: Path,
        cfg: PatchCropConfig,
        provider_spec: PhotoProviderSpec,
        gpu_pool: bool = False,
    ) -> None:
        super().__init__(
            name,
            Path(store_dir),
            deps=[],
            n_unit_workers=cfg.n_unit_workers,
        )
        self._cfg = cfg
        self._provider_spec = provider_spec
        self._provider = provider_spec.build()
        # When True, compute_unit work is dispatched across one process per GPU
        # (see gpu_pool_spec). The coordinator still commits in unit order.
        self._gpu_pool = bool(gpu_pool)
        self._patch_store = PatchStore(self.store_dir / "patches", mode=IOMode.WRITE)
        self._sharpness_store = SharpnessStore(
            self.store_dir / "sharpness.db", mode=IOMode.WRITE
        )
        self._hrng_root = HRng.from_seed(tag_seed(cfg.seed, "crop"))

        self._viz_images_dir: Path | None = None
        if cfg.save_viz_image:
            self._viz_images_dir = self.store_dir / "viz" / "images"
            ensure_directory(self._viz_images_dir)

        self._viz_depths_dir: Path | None = None
        if cfg.save_viz_depth:
            self._viz_depths_dir = self.store_dir / "viz" / "depths"
            ensure_directory(self._viz_depths_dir)

    @property
    def patch_store(self) -> PatchStore:
        return self._patch_store

    def work_units(self) -> list[str]:
        return self._provider.get_names()

    def target_count(self) -> int:
        return self._cfg.n_patches_total

    def gpu_pool_spec(self):
        """Dispatch compute across one process per visible GPU (else None).

        Coordinator still commits in unit order, so output is deterministic
        regardless of which GPU computes a given (photo, cycle).
        """
        if not self._gpu_pool:
            return None
        from datasyn.utils.parallel.mp_per_gpu import get_visible_gpu_ids

        devices = get_visible_gpu_ids()
        if not devices:
            return None
        init_args = (self._provider_spec, self._cfg)
        return (devices, _s1_gpu_init, init_args, _s1_gpu_worker)

    def compute_unit(self, photo: str, cycle: int) -> _PatchCropUnitResult:
        """
        Try n_candidates augmented patches for (photo, cycle).

        Accepted patches are computed only. Store/progress writes happen later
        in commit_unit(), serially and in deterministic photo order.
        Item IDs: "c{cycle:03d}__{photo}__{i:04d}"
        """
        return _compute_patch_crop(
            self._provider, self._cfg, self._hrng_root, photo, cycle
        )

    def commit_unit(self, photo: str, cycle: int, result: _PatchCropUnitResult) -> None:
        del photo, cycle

        target = self.target_count()
        n_done = self.progress.stats()["done"]

        # Select items within the remaining budget, skipping already-done ids.
        to_commit: list[_PatchCropItem] = []
        for item in result.items:
            if n_done >= target:
                break
            if self.progress.is_done(item.patdef.name):
                continue
            to_commit.append(item)
            n_done += 1

        if to_commit:
            # Batched writes: one LMDB txn + one SQLite insert_many per store,
            # instead of per-item commit/fsync (the dominant s1_patch_crop cost).
            self._patch_store.add_many(
                (it.patdef, it.patch_lin_u16, it.patch_dpt) for it in to_commit
            )
            self._sharpness_store.add_many(
                [it.patdef.name for it in to_commit],
                [it.sharpness_raw for it in to_commit],
            )
            self.progress.mark_done_many([it.patdef.name for it in to_commit])

            for item in to_commit:
                item_id = item.patdef.name
                if self._viz_images_dir is not None and item.viz_rgb_u8 is not None:
                    import PIL.Image

                    PIL.Image.fromarray(item.viz_rgb_u8).save(
                        self._viz_images_dir / f"{item_id}.jpg",
                        format="JPEG",
                        quality=90,
                    )

                if self._viz_depths_dir is not None:
                    from datasyn.synthesis.patch_define import (
                        mldepthpro_depth2rgb,
                    )
                    from datasyn.utils.img.convert import arr255u8_to_pil

                    arr255u8_to_pil(mldepthpro_depth2rgb(item.patch_dpt)).save(
                        self._viz_depths_dir / f"{item_id}.png"
                    )
