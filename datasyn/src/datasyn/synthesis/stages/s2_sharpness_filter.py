"""
Stage 2 (s2_sharpness_filter) sharpness filter.

Consumes candidate patches produced by Stage 1 (s1_patch_crop), reproduces the old global
multi-scale Sobel ranking, and writes the selected patch ids as its own stage
result. Downstream stages then depend on this explicit selected-id result
instead of relying on skip semantics in the upstream patch stage.
"""

from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from numpy.typing import NDArray

import numpy as np

from datasyn.pipeline import ItemProgress, StageNode
from datasyn.synthesis.common import IOMode
from datasyn.synthesis.stores.patch_selection_store import PatchSelectionStore
from datasyn.synthesis.stores.sharpness_store import (
    DEFAULT_SHARPNESS_KERNEL_SIZES,
    SharpnessStore,
)

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)


def compute_sharpness_raw_scores_batch(
    patches_rgb: Sequence[np.ndarray],
    kernel_sizes: tuple[int, ...] = DEFAULT_SHARPNESS_KERNEL_SIZES,
) -> np.ndarray:
    """Return raw per-kernel Sobel sharpness sums for a batch of RGB patches."""
    import jax.numpy as jnp

    if not patches_rgb:
        return np.empty((0, len(kernel_sizes)), dtype=np.float32)

    arr = np.asarray(patches_rgb, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected (N, H, W, 3) patches, got {arr.shape}")
    if float(arr.max(initial=0.0)) > 1.5:
        arr = arr / 255.0

    gray = np.einsum(
        "nhwc,c->nhw",
        arr,
        np.array([0.299, 0.587, 0.114], dtype=np.float32),
    )

    # Pad the batch to a power-of-2 size so the conv runs on a small, fixed set
    # of shapes. The batch dim N (number of candidate patches) otherwise varies
    # per call, forcing a fresh XLA compile / cuDNN autotune every time — the
    # dominant s1_patch_crop cost. Bucketing bounds distinct shapes to a handful.
    n = gray.shape[0]
    n_pad = 1
    while n_pad < n:
        n_pad *= 2
    if n_pad != n:
        gray = np.pad(gray, ((0, n_pad - n), (0, 0), (0, 0)))

    compute_scores = _make_batch_sharpness_fn(kernel_sizes)
    scores = np.asarray(compute_scores(jnp.asarray(gray)), dtype=np.float32)
    return scores[:n]


def combine_sharpness_scores(raw_scores: np.ndarray) -> np.ndarray:
    raw_scores = np.asarray(raw_scores, dtype=np.float32)
    if raw_scores.ndim != 2:
        raise ValueError(f"Expected 2-D raw score array, got {raw_scores.shape}")
    s_min = raw_scores.min(axis=0, keepdims=True)
    s_max = raw_scores.max(axis=0, keepdims=True)
    denom = np.where(s_max != s_min, s_max - s_min, 1.0)
    scores_norm = (raw_scores - s_min) / denom
    return np.prod(scores_norm, axis=1, dtype=np.float32)


@dataclass
class SharpnessFilterConfig:
    n_keep: int


class SharpnessFilterNode(StageNode):
    """
    Stage after Stage 1 (s1_patch_crop) candidate generation that materializes selected ids.

    This stage waits for Stage 1 (s1_patch_crop) to finish, ranks the whole candidate pool once,
    stores the kept patch ids under its own store directory, and then exposes
    those ids as its own done-item set for downstream stages.
    """

    def __init__(
        self,
        name: str,
        store_dir: Path,
        cfg: SharpnessFilterConfig,
        deps: list[StageNode],
    ) -> None:
        super().__init__(
            name,
            Path(store_dir),
            deps=deps,
        )
        if len(deps) != 1:
            raise ValueError(
                "SharpnessFilterNode expects exactly one upstream Stage 1 (s1_patch_crop) dep."
            )
        self._cfg = cfg
        self._sharpness_store_path = deps[0].store_dir / "sharpness.db"
        self._selection_store = PatchSelectionStore(
            self.store_dir / "selected.db", mode=IOMode.WRITE
        )
        self._selection_complete_path = self.store_dir / "selection_complete.json"
        self._recover_if_previous_run_was_incomplete()

    def process(self, item_id: str) -> None:
        """No-op: ready_items already exposes only ids selected by this stage."""
        return None

    def ready_items(self) -> list[str]:
        if self.deps and not all(d.is_complete() for d in self.deps):
            return []
        self._ensure_selection_materialized()
        already_handled = self.progress.processable()
        return sorted(set(self._selection_store.list_names()) - already_handled)

    def progress_total(self) -> int | None:
        if self.deps and not all(d.is_complete() for d in self.deps):
            return None
        self._ensure_selection_materialized()
        return len(self._selection_store)

    def _recover_if_previous_run_was_incomplete(self) -> None:
        if self._selection_complete_path.exists():
            if len(self._selection_store) == 0 and self.deps[0].progress.done_items():
                log.warning(
                    "[%s] Found legacy sharpness-filter completion marker without "
                    "selected-id store; resetting this stage to recompute explicit "
                    "selection output.",
                    self.name,
                )
                self._selection_complete_path.unlink(missing_ok=True)
            else:
                return
        if (
            not self.progress.processable()
            and not self.progress.failed_items()
            and len(self._selection_store) == 0
        ):
            return
        log.warning(
            "[%s] Found partial sharpness-filter state without completion marker; "
            "resetting selection store and stage progress.",
            self.name,
        )
        self._selection_store.clear()
        progress_dir = self.store_dir / "_progress"
        for pattern in ("*.db", "*.db-wal", "*.db-shm"):
            for path in progress_dir.glob(pattern):
                path.unlink(missing_ok=True)
        self.progress = ItemProgress(progress_dir)

    def _ensure_selection_materialized(self) -> None:
        if self._selection_complete_path.exists():
            return

        candidate_ids = sorted(self.deps[0].progress.done_items())
        score_store = SharpnessStore(self._sharpness_store_path, mode=IOMode.READ)
        raw_scores = score_store.load_many(candidate_ids)
        combined_scores = combine_sharpness_scores(raw_scores)

        n_keep = min(self._cfg.n_keep, len(candidate_ids))
        if n_keep <= 0:
            keep_names: list[str] = []
        elif n_keep >= len(candidate_ids):
            keep_names = list(candidate_ids)
        else:
            keep_order = np.argsort(combined_scores, kind="stable")[-n_keep:]
            keep_names = sorted(candidate_ids[i] for i in keep_order.tolist())

        self._selection_store.clear()
        self._selection_store.update(keep_names)

        summary = {
            "n_candidates": len(candidate_ids),
            "n_keep": n_keep,
            "kernel_sizes": list(DEFAULT_SHARPNESS_KERNEL_SIZES),
        }
        self._selection_complete_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        log.info(
            "[%s] sharpness filter selected %d / %d candidate patches",
            self.name,
            n_keep,
            len(candidate_ids),
        )


# Multi-scale Sobel sharpness kernels
_KERNEL_SIZES_DEFAULT: tuple[int, ...] = (3, 7, 11, 15)


def _build_sobel_kernels_np(
    kernel_sizes: tuple[int, ...],
) -> list[tuple[NDArray[np.float32], NDArray[np.float32]]]:
    """
    Build 2D Sobel kernel pairs (Gx, Gy) as float32 numpy arrays, exactly
    matching the kernels cv2.Sobel uses internally.

    cv2.getDerivKernels(1, 0, k) returns two 1D kernels:
      kx — derivative kernel (acts along x / columns)
      ky — smoothing kernel  (acts along y / rows)
    The 2D kernel is their outer product: Gx[row, col] = ky[row] * kx[col].
    Gy (y-gradient) is simply Gx.T, by symmetry of the separable construction.
    """
    import cv2

    kernels = []
    for k in kernel_sizes:
        kx, ky = cv2.getDerivKernels(1, 0, k, normalize=False)
        kx = kx.astype(np.float32).flatten()  # (k,) derivative direction
        ky = ky.astype(np.float32).flatten()  # (k,) smooth direction
        Gx = np.outer(ky, kx)  # (k, k)
        Gy = Gx.T  # (k, k)
        kernels.append((Gx, Gy))
    return kernels


@functools.lru_cache(maxsize=8)
def _make_batch_sharpness_fn(
    kernel_sizes: tuple[int, ...] = _KERNEL_SIZES_DEFAULT,
):
    """
    Build and return a jax.jit-compiled function that scores a batch of
    grayscale patches with Sobel sharpness.

    Cached by kernel_sizes so JIT compilation happens only once per unique
    kernel configuration.

    Returns:
        compute_scores: (N, H, W) float32 -> (N, len(kernel_sizes)) float32
            Each column is the summed ||∇I||₂ for one kernel size.
    """
    import jax
    import jax.numpy as jnp

    from datasyn.jaxutils import jx

    kernels_np = _build_sobel_kernels_np(kernel_sizes)
    # Convert to JAX float32 arrays; closed over in the jitted function so
    # they are treated as compile-time constants by XLA.
    kernels_jax = [
        (jnp.array(Gx, dtype=jnp.float32), jnp.array(Gy, dtype=jnp.float32))
        for Gx, Gy in kernels_np
    ]

    @jx.jit
    def compute_scores(gray_batch: jax.Array) -> jax.Array:
        """
        Args:
            gray_batch: (N, H, W) float32
        Returns:
            (N, num_kernels) float32
        """
        lhs = gray_batch[:, None, :, :]  # (N, 1, H, W) — NCHW for conv

        scores_list = []
        # Static Python loop: unrolled at trace time, one XLA conv per kernel size.
        for (Gx, Gy), k in zip(kernels_jax, kernel_sizes):
            # Stack Gx and Gy as a 2-output-channel filter: (2, 1, k, k) OIHW
            filters = jnp.stack([Gx, Gy])[:, None, :, :]
            pad = k // 2
            out = jax.lax.conv_general_dilated(
                lhs,
                filters,
                window_strides=(1, 1),
                padding=((pad, pad), (pad, pad)),
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            )  # (N, 2, H, W)
            # Sum hypot(gx, gy) over spatial dims -> (N,)
            score = jnp.sum(jnp.hypot(out[:, 0], out[:, 1]), axis=(-2, -1))
            scores_list.append(score)

        return jnp.stack(scores_list, axis=1)  # (N, num_kernels)

    return compute_scores
