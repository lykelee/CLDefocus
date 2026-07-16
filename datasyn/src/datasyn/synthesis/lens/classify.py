from __future__ import annotations

import logging
import math
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasyn.pipeline import StageNode
from datasyn.synthesis.subprocess_nodes import (
    SubprocessStageRunNode,
    cpu_only_jax_env,
)
from datasyn.utils.sqlite.tablestore import SQLiteTableStore

log = logging.getLogger(__name__)


def classify_lens(efl_mm: float, fnum: float, imgrad_mm: float, ttl_mm: float):
    no_class = "Unknown", "N/A"

    if fnum > 8.0:
        return no_class

    hfov = math.degrees(math.atan(imgrad_mm / efl_mm))
    tele_ratio = ttl_mm / efl_mm

    if (2.5 <= imgrad_mm <= 8.0) and (ttl_mm <= 15.0):
        category = "Mobile"
        if hfov > 40:
            return category, "Ultra-Wide"
        if hfov > 30:
            return category, "Wide (Main)"
        if hfov > 10:
            return category, "Telephoto"
        return category, "Super-Tele (Periscope)"

    elif (imgrad_mm >= 10.0) and (ttl_mm >= 25.0):
        category = "System Camera"
        if hfov > 35:
            if tele_ratio > 1.2:
                return category, "Retrofocus Wide"
            return category, "Wide"
        elif 18 <= hfov <= 35:
            return category, "Standard/Normal"
        elif hfov < 18:
            if tele_ratio < 1.0:
                return category, "Compact Telephoto"
            return category, "Telephoto"

    return no_class


CLASSIFY_COLUMNS: dict[str, type] = {
    "name": str,
    "file": str,
    "category": str,
    "subcategory": str,
}


class ClassifyLensNode(StageNode):
    """Classify lenses by paraxial/geometry features into a SQLite table."""

    def __init__(
        self,
        name: str,
        store_dir: Path,
        deps: list[StageNode],
        lens_root: Path,
        classify_db_path: Path,
        n_workers: int = 8,
    ) -> None:
        super().__init__(name, store_dir, deps=deps, n_workers=1)
        self._lens_root = Path(lens_root)
        self._classify_db_path = Path(classify_db_path)
        self._n_workers = n_workers

    def source_items(self) -> list[str]:
        return sorted([str(p) for p in self._lens_root.rglob("*.mytable")])

    def process(self, item_id: str) -> None:  # noqa: ARG002
        raise NotImplementedError(
            "ClassifyLensNode dispatches via its own thread pool."
        )

    def _process_batch(self, item_ids: list[str], fail_fast: bool) -> None:
        from datasyn.jaxutils.configs import easy_optics_setup

        easy_optics_setup()

        import jax.numpy as jnp

        from datasyn.optics.complens.tabular_lens import load_tabular_lens_from_mytable
        from datasyn.optics.ray import DEFAULT_WAVELENGTH

        store = SQLiteTableStore(
            self._classify_db_path, CLASSIFY_COLUMNS, primary_key="name"
        )

        def _run_one(item_id: str) -> tuple[str, dict | None, str | None]:
            try:
                lens_file = Path(item_id)
                lens_name = lens_file.stem

                lens = load_tabular_lens_from_mytable(lens_file)

                wvln = DEFAULT_WAVELENGTH
                rtm, ep, xp = lens.calc_paraxial_properties(wvln)

                ttl = jnp.abs(rtm.ttl())
                efl = jnp.abs(rtm.efl())
                fnum = efl / (2 * ep.r)
                imgrad = lens.imgrad

                category, subcategory = classify_lens(
                    efl_mm=float(1e3 * efl),
                    fnum=float(fnum),
                    imgrad_mm=float(1e3 * imgrad),
                    ttl_mm=float(1e3 * ttl),
                )

                record = {
                    "name": lens_name,
                    "file": str(lens_file),
                    "category": str(category),
                    "subcategory": str(subcategory),
                }
                return item_id, record, None
            except Exception:
                return item_id, None, traceback.format_exc()

        failed: list[str] = []
        with ThreadPoolExecutor(max_workers=self._n_workers) as ex:
            futures = {ex.submit(_run_one, i): i for i in item_ids}
            for f in as_completed(futures):
                item_id, record, err = f.result()
                if err is not None:
                    self.progress.mark_failed(item_id, err)
                    failed.append(item_id)
                    log.warning("[%s] %s failed:\n%s", self.name, item_id, err)
                    if fail_fast:
                        raise RuntimeError(f"[{self.name}] {item_id} failed:\n{err}")
                else:
                    store.delete("name", record["name"])
                    store.insert(record)
                    self.progress.mark_done(item_id)

        store.close()

        log.info(
            "[%s] classify batch: ok=%d failed=%d",
            self.name,
            len(item_ids) - len(failed),
            len(failed),
        )


def _make_classify_lens_node(
    name: str,
    store_dir: Path,
    deps: list[StageNode],
    lens_root: Path,
    classify_db_path: Path,
    n_workers: int,
) -> ClassifyLensNode:
    return ClassifyLensNode(
        name,
        store_dir,
        deps=deps,
        lens_root=lens_root,
        classify_db_path=classify_db_path,
        n_workers=n_workers,
    )


def make_classify_subprocess_node(
    name: str,
    store_dir: Path,
    deps: list[StageNode],
    lens_root: Path,
    classify_db_path: Path,
    n_workers: int = 8,
) -> SubprocessStageRunNode:
    Path(classify_db_path).parent.mkdir(parents=True, exist_ok=True)
    SQLiteTableStore(classify_db_path, CLASSIFY_COLUMNS, primary_key="file").close()

    root_total: int | None
    if not deps:
        root_total = sum(1 for _ in Path(lens_root).rglob("*.mytable"))
    else:
        root_total = None

    return SubprocessStageRunNode(
        name,
        store_dir,
        deps=deps,
        factory=_make_classify_lens_node,
        factory_args=(Path(lens_root), Path(classify_db_path), n_workers),
        child_env=cpu_only_jax_env(),
        root_total=root_total,
    )
