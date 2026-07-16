from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from datasyn.pipeline import StageNode
from datasyn.utils.sqlite.tablestore import SQLiteTableStore

if TYPE_CHECKING:
    from datasyn.utils.parallel.mp_per_gpu import Context

log = logging.getLogger(__name__)


STATS_COLUMNS: dict[str, type] = {
    "name": str,
    "file": str,
    "valid": bool,
    "crit": bool,
    "power": float,
    "fpp": float,
    "bpp": float,
    "efl": float,
    "ttl": float,
    "epz": float,
    "epr": float,
    "xpz": float,
    "xpr": float,
    "na": float,
    "hfov": float,
    "imgrad": float,
    "rmse_field0": float,
    "phase_diff_field0": float,
    "phase_diff_field1": float,
}


class _StatsGPUData(NamedTuple):
    stats_db_path: str
    lens_root: str
    jax_devices: tuple[str, ...]


class _StatsWorkerSummary(NamedTuple):
    item_id: str
    lens_name: str
    ok: bool
    error_tb: str


def _nan_if_none(x: Any) -> float:
    if x is None:
        return float("nan")
    return float(x)


def _stats_init(
    ctx: "Context[_StatsGPUData]",
    lens_root: str,
    stats_db_path: str,
) -> _StatsGPUData:
    from datasyn.jaxutils.configs import easy_optics_setup

    easy_optics_setup()

    import jax

    jax_devices = tuple(str(d) for d in jax.devices())

    log.info(
        "[stats_lens] subprocess init: device=%s CUDA_VISIBLE_DEVICES=%s jax.devices=%s",
        ctx.device,
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        jax_devices,
    )
    return _StatsGPUData(
        stats_db_path=stats_db_path,
        lens_root=lens_root,
        jax_devices=jax_devices,
    )


def _stats_worker(
    ctx: "Context[_StatsGPUData]",
    task_id: str,
) -> _StatsWorkerSummary:
    from functools import partial

    import jax
    import jax.numpy as jnp

    import datasyn.optics.ray as raylib
    from datasyn.jaxutils import jx
    from datasyn.optics.complens.tabular_lens import load_tabular_lens_from_mytable
    from datasyn.optics.imaging.debye import calc_na
    from datasyn.optics.imaging.imaging import make_imaging_helper
    from datasyn.optics.imaging.pupil_function.pf import (
        generate_rayfront,
        wavefront_by_centroid,
    )
    from datasyn.optics.imaging.pupil_function.sampling import (
        lens_best_focus_by_intersection,
    )
    from datasyn.optics.ray import DEFAULT_WAVELENGTH

    lens_file = Path(task_id)
    lens_name = lens_file.stem

    try:
        lens = load_tabular_lens_from_mytable(lens_file)

        wvl = DEFAULT_WAVELENGTH
        rtm, ep, xp = lens.calc_paraxial_properties(wvl)

        na = calc_na(ep.r, rtm.efl())
        tanhfov = lens.imgrad / jnp.abs(rtm.efl())
        imghelp = make_imaging_helper(lens)
        field = imghelp.dist_to_field(100.0)

        hfov_deg = jnp.rad2deg(jnp.arctan(tanhfov))

        rng = jax.random.PRNGKey(0)

        valid = True
        rmse = None
        phase_diffs = [None, None]

        for i, f in enumerate([0.0, 1.0]):
            f_xy = jnp.array([f, 0.0])

            rf = generate_rayfront(
                lens=lens,
                ep=ep,
                rng=rng,
                n=63,
                field=field,
                f_xy=f_xy,
                wvln=wvl,
                verbose=False,
            ).rf

            if jnp.sum(rf.rays.valid) == 0:
                valid = False
                continue

            pt_foc = lens_best_focus_by_intersection(
                lens=lens,
                z_start=-1e0,
                ep=ep,
                rng=rng,
                field=field,
                f_xy=jnp.array([f, 0.0]),
                wvl=wvl,
            )
            wf = wavefront_by_centroid(rf, xp.z, pt_foc=pt_foc).wf
            phase = wf.phase()
            phase_diff = phase.max() - phase.min()
            phase_diffs[i] = phase_diff

            if f == 0.0:
                from datasyn.optics.imaging.pupil_function.sampling import ray_xy_rmse

                rays_img = jx.vmap(partial(raylib.project_to_z, lens.imgpos))(
                    rf.rays
                ).ray
                rmse = ray_xy_rmse(rays_img, jnp.array([0.0, 0.0]))

        if not valid:
            crit = None
        else:
            crit_1 = rmse < 0.01 * lens.imgrad
            crit_2 = na < 0.6
            crit_3 = all([x < (20 * jnp.pi) for x in phase_diffs])
            crit = crit_1 and crit_2 and crit_3

        record = {
            "name": lens_name,
            "file": str(lens_file),
            "valid": bool(valid),
            "crit": bool(crit) if crit is not None else False,
            "power": float(rtm.power()),
            "fpp": float(rtm.fpp()),
            "bpp": float(rtm.bpp()),
            "efl": float(rtm.efl()),
            "ttl": float(rtm.ttl()),
            "epz": float(ep.z),
            "epr": float(ep.r),
            "xpz": float(xp.z),
            "xpr": float(xp.r),
            "na": float(na),
            "hfov": float(hfov_deg),
            "imgrad": float(lens.imgrad),
            "rmse_field0": _nan_if_none(rmse),
            "phase_diff_field0": _nan_if_none(phase_diffs[0]),
            "phase_diff_field1": _nan_if_none(phase_diffs[1]),
        }

        store = SQLiteTableStore(
            ctx.data.stats_db_path, STATS_COLUMNS, primary_key="name"
        )
        store.delete("name", lens_name)
        store.insert(record)

        return _StatsWorkerSummary(
            item_id=task_id, lens_name=lens_name, ok=True, error_tb=""
        )

    except Exception:
        tb = traceback.format_exc()
        log.warning("[stats_lens] lens %r failed:\n%s", lens_name, tb)
        return _StatsWorkerSummary(
            item_id=task_id, lens_name=lens_name, ok=False, error_tb=tb
        )


class StatsLensNode(StageNode):
    """Compute per-lens optical stats across multiple GPUs, row per lens."""

    def __init__(
        self,
        name: str,
        store_dir: Path,
        deps: list[StageNode],
        lens_root: Path,
        stats_db_path: Path,
    ) -> None:
        super().__init__(name, store_dir, deps=deps)
        self._lens_root = Path(lens_root)
        self._stats_db_path = Path(stats_db_path)

        Path(stats_db_path).parent.mkdir(parents=True, exist_ok=True)
        SQLiteTableStore(stats_db_path, STATS_COLUMNS, primary_key="file").close()

    def source_items(self) -> list[str]:
        return sorted([str(p) for p in self._lens_root.rglob("*.mytable")])

    def process(self, item_id: str) -> None:  # noqa: ARG002
        raise NotImplementedError(
            "StatsLensNode dispatches per-lens GPU tasks via _process_batch."
        )

    def _process_batch(self, item_ids: list[str], fail_fast: bool) -> None:
        from datasyn.utils.parallel import MPPerGPUConfig, run_mp_per_gpu_iter

        cfg_mp = MPPerGPUConfig(
            worker_func=_stats_worker,
            indices=sorted(item_ids),
            worker_args=(),
            fn_init_proc=_stats_init,
            init_args=(
                str(self._lens_root),
                str(self._stats_db_path),
            ),
            force_mp=False,
            show_progress=False,
        )

        failed: list[str] = []
        n_ok = 0
        for summary in run_mp_per_gpu_iter(cfg_mp):
            if summary.ok:
                self.progress.mark_done(summary.item_id)
                n_ok += 1
            else:
                self.progress.mark_failed(summary.item_id, summary.error_tb)
                failed.append(summary.item_id)

        log.info(
            "[%s] stats summaries: ok=%d failed=%d",
            self.name,
            n_ok,
            len(failed),
        )

        if failed and fail_fast:
            raise RuntimeError(f"[{self.name}] {len(failed)} lens(es) failed: {failed}")
