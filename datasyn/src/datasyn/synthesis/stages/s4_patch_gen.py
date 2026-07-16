"""
Stage 4 (s4_patch_gen) — PatchGenerateNode

Builds the per-layer depth description of each patch (NOT the image layers).
For each patch:
  1. Load depth from PatchStore (upstream Stage 1 (s1_patch_crop) payload).
  2. Load lens/focusing from CameraStore (upstream Stage 3 (s3_camera)).
  3. Cluster depth by signed CoC to reduce layer count.
  4. Store the per-pixel cluster index map + per-layer depths to LayerStore.

The heavy inpainted RGBA image layers are no longer stored here — Stage 6
(s6_blur) rebuilds them online from a linear patch + the cluster index map, so
they never hit disk.

Reads payloads from: deps[0].store_dir/patches    (PatchStore — Stage 1 (s1_patch_crop))
Reads camera data:   deps[1].store_dir/cameras.db (CameraStore — Stage 3 (s3_camera))
Writes to:           store_dir/layers             (LayerStore, depth-only)

Item ID: same patch name as Stages 1 and 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd

from datasyn.pipeline import StageNode
from datasyn.synthesis.common import IOMode
from datasyn.synthesis.stores.camera_store import CameraStore
from datasyn.synthesis.stores.layer_store import LayerStore
from datasyn.synthesis.stores.patch_store import PatchStore
from datasyn.utils.files import ensure_directory

if TYPE_CHECKING:
    from datasyn.optics.paraxial import Pupil
    from datasyn.optics.paraxial.rtm2 import VirtualThinLens


@dataclass
class PatchGenConfig:
    stats_db_path: Path  # path to structured lens stats DB
    seed: int = 0  # base seed
    save_viz: bool = True  # save clustered-depth images for inspection
    save_coc: bool = True
    # Ablation: remove intra-patch depth variation.
    ablate_vardpt: bool = False


class _LensStats(NamedTuple):
    thinlens: "VirtualThinLens"
    xp: "Pupil"
    imgrad: "float"


class PatchGenerateNode(StageNode):
    """
    Downstream stage: builds layered depth scenes.

    Dependency contract:
    - `deps[0]` must be the raw patch-producing stage so this node can read
      `PatchStore`.
    - `deps[1]` must be the camera-selection stage that already reflects any
      upstream patch filtering in its ready-item set.
    - Additional deps, if any, are treated as gating only and are not read
      directly by this node.

    store_dir layout::

        store_dir/
            layers/         <- LayerStore
            viz/depths/     <- clustered-depth JPEG images (when save_viz=True)
            _progress/      <- ItemProgress
    """

    def __init__(
        self,
        name: str,
        store_dir: Path,
        cfg: PatchGenConfig,
        deps: list[StageNode],
        n_workers: int = 1,
    ) -> None:
        if len(deps) < 2:
            raise ValueError(
                "PatchGenerateNode requires deps[0] = patch stage and "
                "deps[1] = camera stage"
            )

        super().__init__(name, Path(store_dir), deps=deps, n_workers=n_workers)
        self._cfg = cfg

        # Open upstream stores (read-only).
        # TODO 260508: Issue! Stores may not be generated yet at the first run
        #              This raises error!
        self._patch_store = PatchStore(deps[0].store_dir / "patches", mode=IOMode.READ)
        self._camera_store = CameraStore(
            deps[1].store_dir / "cameras.db", mode=IOMode.READ
        )
        # Open own LayerStore (write).
        self._layer_store = LayerStore(store_dir / "layers", mode=IOMode.WRITE)

        self._viz_depths_dir: Path | None = None
        if cfg.save_viz:
            self._viz_depths_dir = store_dir / "viz" / "depths"
            ensure_directory(self._viz_depths_dir)

        self._coc_dir: Path | None = None
        if cfg.save_coc:
            self._coc_dir = store_dir / "coc"
            ensure_directory(self._coc_dir)

        # Build lens name -> stats dict (O(n) over the CSV).
        import jax.numpy as jnp

        from datasyn.optics.paraxial import Pupil
        from datasyn.optics.paraxial.rtm2 import VirtualThinLens
        from datasyn.synthesis.lens.stats import STATS_COLUMNS
        from datasyn.utils.sqlite.tablestore import SQLiteTableStore

        stats_store = SQLiteTableStore(
            cfg.stats_db_path,
            STATS_COLUMNS,
            primary_key="file",
            readonly=True,
        )
        rows = stats_store.find()
        stats_store.close()
        df = pd.DataFrame(rows)
        if len(df) == 0:
            raise ValueError(f"No lens rows found in '{cfg.stats_db_path}'")
        self._lens_stats: dict[str, _LensStats] = {}
        for _, row in df.iterrows():
            self._lens_stats[row["file"]] = _LensStats(
                thinlens=VirtualThinLens(
                    power=jnp.asarray(row["power"]),
                    fpp=jnp.asarray(row["fpp"]),
                    bpp=jnp.asarray(row["bpp"]),
                ),
                xp=Pupil(z=jnp.asarray(row["xpz"]), r=jnp.asarray(row["xpr"])),
                imgrad=float(row["imgrad"]),
            )

    # ------------------------------------------------------------------

    def process(self, item_id: str) -> None:
        import jax.numpy as jnp
        import jax.random as jr

        from datasyn.jaxutils import jx
        from datasyn.jaxutils.random import HRng, tag_seed
        from datasyn.synthesis.camera_choose import (
            calc_obj_from_signed_coc,
            calc_signed_coc,
            safe_cluster,
        )
        from datasyn.utils.img.convert import arr1f_to_pil, arr255u8_to_pil

        patdef = self._patch_store.load_patdef(item_id)
        depth = self._patch_store.load_depth(item_id)
        cam = self._camera_store.load(item_id)
        ls = self._lens_stats[cam.lens]
        thinlens, xp, focusing = ls.thinlens, ls.xp, cam.focusing

        # Cluster depth by signed CoC to reduce layer count.
        depth_flat = depth.flatten()

        def signed_coc(dpt_obj):
            return calc_signed_coc(thinlens, xp, -dpt_obj, -focusing)

        def scoc_to_dpt(scoc):
            val = -1.0 * calc_obj_from_signed_coc(thinlens, xp, scoc, -focusing)
            val = jnp.where(
                jnp.isneginf(val), jnp.inf, val
            )  # The above formula may give -inf, which must be replaced with +inf.
            val = jnp.clip(val, patdef.depth_min, patdef.depth_max)
            return val

        photo_hw = patdef.photo_size
        pixsize = ls.imgrad * (2.0 / np.sqrt(photo_hw[0] ** 2 + photo_hw[1] ** 2))

        clst_eps = 0.5 * pixsize

        # Always compute the per-pixel signed CoC (also used by the CoC viz).
        scoc_arr = jx.vmap(signed_coc)(depth_flat)

        if self._cfg.ablate_vardpt:
            # Ablation: pick one random pixel's depth and fill the whole patch
            # with it -> a single uniform depth layer. RNG is derived from the
            # base seed + item_id so the pick is reproducible and independent of
            # scheduling (same HRng pattern as the other stochastic stages).
            key = HRng.from_seed(tag_seed(self._cfg.seed, "ablate vardpt"))[item_id].key
            idx_dpt = int(jr.randint(key, (), 0, depth_flat.shape[0]))
            dd = float(depth_flat[idx_dpt])
            dd = float(np.clip(dd, patdef.depth_min, patdef.depth_max))
            clst_dpt = np.full(depth.shape, dd, dtype=np.float64)
        else:
            clst_out = safe_cluster(scoc_arr.reshape(*depth.shape), eps=clst_eps)
            clst_scoc = clst_out.mapped
            clst_dpt = np.asarray(jx.vmap(scoc_to_dpt)(clst_scoc))

        # Store ONLY the per-pixel cluster index map + per-layer depths; Stage 6
        # (s6_blur) rebuilds the inpainted image layers online (RAM, no disk). The
        # integer index map (not quantized depths) keeps the layer count/order
        # identical to what Stage 5 (s5_psf_gen) consumes from l2d.
        levels, idx_inv = np.unique(clst_dpt, return_inverse=True)
        idx_map = idx_inv.reshape(clst_dpt.shape).astype(np.uint16)
        self._layer_store.add_depth_layers(item_id, idx_map, levels, patdef.photo)

        if self._viz_depths_dir is not None:
            from datasyn.synthesis.patch_define import mldepthpro_depth2rgb

            arr255u8_to_pil(mldepthpro_depth2rgb(np.asarray(clst_dpt))).save(
                self._viz_depths_dir / f"{item_id}.png"
            )

        if self._coc_dir is not None:
            coc_map = jnp.abs(jnp.asarray(scoc_arr)).reshape(*depth.shape)
            coc_rgb = jnp.clip(coc_map / (pixsize * 64), 0, 1)
            arr1f_to_pil(coc_rgb).save(self._coc_dir / f"{item_id}.png")
