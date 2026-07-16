"""
StageNode: the unit of work in a persistent data-parallel DAG pipeline.
"""

from __future__ import annotations

import logging
import threading
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .progress import ItemProgress

log = logging.getLogger(__name__)


class StageNode(ABC):
    """
    A single stage in a persistent data-parallel DAG pipeline.

    Subclass and implement ``process(item_id)``.
    Root nodes (no deps) must also implement ``source_items()``.

    Layout on disk::

        store_dir/
            _progress/
                done.db    <- SetStore: successfully completed items
                skipped.db <- SetStore: intentionally excluded items
                errors.db  <- KVStore :  item_id ->latest traceback
            <user data>    <- anything the stage writes; up to the subclass

    Item lifecycle::

        pending -> (process called) -> done   : propagates downstream
                                    -> skipped: stops here, not propagated
                                    -> failed : retried on the next run

    Resumability:
        Done and skipped items are never re-processed.
        Failed items are retried on every run until they succeed or are skipped.

    Thread safety:
        If ``n_workers > 1``, multiple items are processed concurrently.
        Ensure ``process()`` is thread-safe in that case.
    """

    def __init__(
        self,
        name: str,
        store_dir: Path,
        deps: list[StageNode] | None = None,
        n_workers: int = 1,
        exclusive_lock: threading.Lock | None = None,
    ) -> None:
        self.name = name
        self.store_dir = Path(store_dir)
        self.deps: list[StageNode] = deps or []
        self.n_workers = n_workers
        self.progress = ItemProgress(self.store_dir / "_progress")
        self._exclusive_lock = exclusive_lock

    # -------------------------------------------------------------------------
    # User API — implement these in subclasses
    # -------------------------------------------------------------------------

    @abstractmethod
    def process(self, item_id: str) -> None:
        """
        Process a single item. Called by the framework.

        Outcomes:
            - Return normally -> item is marked **done**; propagates downstream.
            - Call ``self.progress.mark_skipped(item_id)`` -> item is **skipped**;
              does not propagate downstream (e.g., quality filter rejection).
            - Raise any exception -> item is **failed**; retried on next run.

        Conventions:
            - Read upstream data from ``self.deps[i].store_dir``.
            - Write output data under ``self.store_dir``.
        """
        ...

    def source_items(self) -> list[str]:
        """
        Root nodes only (no deps): return all item IDs this stage should process.

        Called on each ``ready_items()`` poll, so this should be cheap.
        If generating the list is expensive, precompute it in ``__init__``
        and return the cached result here.

        Not called for non-root nodes (they derive items from their deps).
        """
        raise NotImplementedError(
            f"Node '{self.name}' has no deps and must implement source_items()."
        )

    # -------------------------------------------------------------------------
    # Framework — no need to override
    # -------------------------------------------------------------------------

    def ready_items(self) -> list[str]:
        """
        Items that are ready to process right now:

        - For root nodes: items in ``source_items()`` not yet handled.
        - For non-root nodes: items marked **done** by ALL upstream deps,
          that this stage has not yet handled.

        Note: only upstream ``done`` items propagate here.
        Upstream ``skipped`` items stop at their stage and are not included.
        """
        already_handled = self.progress.processable()

        if not self.deps:
            return sorted(set(self.source_items()) - already_handled)

        dep_done_sets = [d.progress.done_items() for d in self.deps]
        upstream_done = dep_done_sets[0].intersection(*dep_done_sets[1:])
        return sorted(upstream_done - already_handled)

    def is_complete(self) -> bool:
        """
        True when this stage has nothing left to do and never will.

        - Root nodes: all ``source_items()`` are handled (done or skipped).
        - Non-root nodes: all upstream deps are complete AND no ready items remain.
        """
        if not self.deps:
            return len(self.ready_items()) == 0
        return all(d.is_complete() for d in self.deps) and len(self.ready_items()) == 0

    def run(
        self,
        fail_fast: bool = False,
    ) -> None:
        """
        Execute this stage: process all currently ready items once, then return.

        Args:
            fail_fast:
                If ``True``, let the first item-level error propagate out
                untouched (no try/except around ``process()``) so IDE debuggers
                break at the real failure site with live frames and locals.
                In this mode the failing item is **not** recorded in
                ``errors.db`` — the trade-off for a usable stack trace.
                With ``n_workers > 1``, the exception surfaces via
                ``Future.result()`` after sibling workers in the batch finish.
        """
        ready = self.ready_items()

        if ready:
            log.debug(f"[{self.name}] Processing {len(ready)} item(s).")
            if self._exclusive_lock is not None:
                with self._exclusive_lock:
                    self._process_batch(ready, fail_fast=fail_fast)
            else:
                self._process_batch(ready, fail_fast=fail_fast)

        log.debug(f"[{self.name}] Complete. {self.progress}")

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _process_batch(self, item_ids: list[str], fail_fast: bool) -> None:
        if self.n_workers <= 1:
            for item_id in item_ids:
                self._run_one(item_id, fail_fast=fail_fast)
        else:
            with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
                futures = {
                    ex.submit(self._run_one, item_id, fail_fast): item_id
                    for item_id in item_ids
                }
                for f in as_completed(futures):
                    # Propagates the re-raise from _run_one when fail_fast=True.
                    # When fail_fast=False, _run_one never raises, so this is a no-op.
                    f.result()

    def _run_one(self, item_id: str, fail_fast: bool = False) -> None:
        if fail_fast:
            # Do not wrap in try/except: once an except clause catches the
            # exception the original frames are unwound, so IDE debuggers
            # (VS Code, pdb) cannot inspect locals at the real failure site.
            # Let it propagate untouched through _process_batch -> run() ->
            # Pipeline.run_batch() so "break on uncaught exception" fires at
            # the actual process() call. Trade-off: this item is not recorded
            # in errors.db — fine for interactive debugging.
            self.process(item_id)
            if not self.progress.is_skipped(item_id):
                self.progress.mark_done(item_id)
            return
        try:
            self.process(item_id)
            # Mark done only if process() didn't call mark_skipped().
            if not self.progress.is_skipped(item_id):
                self.progress.mark_done(item_id)
        except Exception:
            tb = traceback.format_exc()
            self.progress.mark_failed(item_id, tb)
            log.warning(f"[{self.name}] Item '{item_id}' failed:\n{tb}")

    def __repr__(self) -> str:
        deps_str = ", ".join(d.name for d in self.deps)
        return f"StageNode(name={self.name!r}, deps=[{deps_str}])"


class GroupedStageNode(StageNode):
    """
    A StageNode that collects items into groups before processing.

    Override ``group_key()`` and ``process_group()`` instead of ``process()``.
    Items sharing the same group key are dispatched together to ``process_group()``,
    allowing shared setup work (e.g., loading a large file once per group).

    Item lifecycle inside ``process_group()``:
        - Return normally -> items not yet marked skipped/done are marked **done**.
        - Call ``self.progress.mark_skipped(item_id)`` -> item is **skipped**.
        - Raise an exception -> all unfinished items in the group are marked **failed**.

    Threading:
        If ``n_workers > 1``, groups are processed concurrently (one thread per group).
        Ensure ``process_group()`` is thread-safe in that case.
    """

    @abstractmethod
    def group_key(self, item_id: str) -> str:
        """Map an item_id to its group key.  Items with the same key are batched."""
        ...

    @abstractmethod
    def process_group(self, group_key: str, item_ids: list[str]) -> None:
        """
        Process a batch of items that share the same group key.

        The framework marks items done/skipped/failed automatically — see class
        docstring for the outcome rules.
        """
        ...

    def process(self, item_id: str) -> None:  # noqa: ARG002
        raise NotImplementedError(
            f"GroupedStageNode '{self.name}' uses process_group(), not process()"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_batch(self, item_ids: list[str], fail_fast: bool) -> None:
        groups: dict[str, list[str]] = defaultdict(list)
        for item_id in item_ids:
            groups[self.group_key(item_id)].append(item_id)

        if self.n_workers <= 1:
            for key, ids in groups.items():
                self._run_group(key, ids, fail_fast=fail_fast)
        else:
            with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
                futures = {
                    ex.submit(self._run_group, key, ids, fail_fast): key
                    for key, ids in groups.items()
                }
                for f in as_completed(futures):
                    f.result()

    def _run_group(self, key: str, ids: list[str], fail_fast: bool) -> None:
        if fail_fast:
            # See StageNode._run_one for rationale: no try/except so debuggers
            # can break at the real failure site inside process_group().
            self.process_group(key, ids)
            for item_id in ids:
                if not self.progress.is_skipped(item_id) and not self.progress.is_done(
                    item_id
                ):
                    self.progress.mark_done(item_id)
            return
        try:
            self.process_group(key, ids)
            for item_id in ids:
                if not self.progress.is_skipped(item_id) and not self.progress.is_done(
                    item_id
                ):
                    self.progress.mark_done(item_id)
        except Exception:
            tb = traceback.format_exc()
            log.warning(f"[{self.name}] Group '{key}' failed:\n{tb}")
            for item_id in ids:
                if not self.progress.is_done(item_id) and not self.progress.is_skipped(
                    item_id
                ):
                    self.progress.mark_failed(item_id, tb)


class WhileSourceNode(StageNode):
    """
    A source stage that cycles over a fixed work-unit list until a target
    number of items have been produced.

    Override ``work_units()``, ``target_count()``, and ``process_unit()``.

    Execution model
    ---------------
    The node loops over ``work_units()`` repeatedly (cycle 0, 1, 2, ...).
    For each ``(unit, cycle)`` pair it calls ``process_unit(unit, cycle)``,
    which stores produced items and calls ``self.progress.mark_done(item_id)``
    for each one.  The loop ends when ``len(done_items()) >= target_count()``.

    Determinism
    -----------
    ``(unit, cycle)`` is passed to ``process_unit`` so the subclass can seed a
    fully deterministic JAX PRNGKey per call.  Item IDs must encode both to
    guarantee they are stable across restarts (e.g., ``"c001__photo__0003"``).

    Resumability
    ------------
    A small checkpoint DB (``_progress/cycle_checkpoint.db``) stores the
    ``(cycle, unit_idx)`` of the last successfully completed unit.  On restart
    the loop skips already-completed units and recomputes ``produced`` from the
    persistent ``done.db``.  ``process_unit`` must be idempotent: if called
    again for the same ``(unit, cycle)`` after a crash, it must not
    double-count already-stored items (``mark_done`` and typical store writes
    are both idempotent).

    Progress display
    ----------------
    ``progress.stats()["done"]`` = items produced so far.
    ``progress_total()``         = ``target_count()``  (used for tqdm bar).

    Downstream compatibility
    ------------------------
    ``progress.done_items()`` returns the set of produced item IDs, which
    downstream ``StageNode`` instances consume via their ``ready_items()``
    exactly as they would from any other upstream stage.
    """

    def __init__(
        self,
        name: str,
        store_dir: Path,
        deps: list[StageNode] | None = None,
        max_consecutive_unit_failures: int = 5,
    ) -> None:
        # n_workers is not used by WhileSourceNode (the cycling loop is
        # inherently sequential), but StageNode requires it — pass 1.
        super().__init__(name, store_dir, deps=deps or [], n_workers=1)
        # Lazy import: mysqliteutils is already a transitive dep (via progress.py).
        from datasyn.utils.sqlite.kvstore import KVStore

        # _progress/ is already created by ItemProgress inside super().__init__
        self._ckpt = KVStore(self.store_dir / "_progress" / "cycle_checkpoint.db")
        self._max_consecutive_unit_failures = max_consecutive_unit_failures
        # In-memory only: how many *consecutive* cycles a given unit has just
        # failed in. Reset on success. Not persisted across restarts.
        self._unit_fail_streak: dict[str, int] = {}

    # -------------------------------------------------------------------------
    # Abstract interface — implement these in subclasses
    # -------------------------------------------------------------------------

    @abstractmethod
    def work_units(self) -> list[str]:
        """
        Fixed ordered list of work units to cycle over (e.g., photo names).

        Called once at the start of ``run()`` and cached.  Must be cheap.
        """
        ...

    @abstractmethod
    def target_count(self) -> int:
        """Total number of items to produce across all cycles."""
        ...

    @abstractmethod
    def process_unit(self, unit: str, cycle: int) -> None:
        """
        Process one ``(unit, cycle)`` pair.

        Store all produced items and call ``self.progress.mark_done(item_id)``
        for each.  The item IDs must encode both ``unit`` and ``cycle`` so they
        are stable and unique across all cycles (e.g., ``"c001__photo__0003"``).

        This method must be **idempotent**: if called again for the same
        ``(unit, cycle)`` after a crash, it must produce exactly the same items
        without duplicating progress records (``mark_done`` is a set-add;
        typical store writes are insert-or-ignore).
        """
        ...

    # -------------------------------------------------------------------------
    # Framework overrides
    # -------------------------------------------------------------------------

    def progress_total(self) -> int:
        """Target count, used by Pipeline._compute_total for tqdm bar."""
        return self.target_count()

    def is_complete(self) -> bool:
        return len(self.progress.done_items()) >= self.target_count()

    def source_items(self) -> list[str]:
        raise NotImplementedError(
            f"WhileSourceNode '{self.name}' does not use source_items(); "
            "use work_units() and target_count() instead."
        )

    def process(self, item_id: str) -> None:
        raise NotImplementedError(
            f"WhileSourceNode '{self.name}' uses process_unit(), not process()"
        )

    def run(
        self,
        fail_fast: bool = False,
    ) -> None:
        units = self.work_units()
        if not units:
            log.warning(f"[{self.name}] work_units() is empty — nothing to do.")
            return

        cycle, unit_start = self._load_checkpoint()
        produced = len(self.progress.done_items())

        log.info(
            f"[{self.name}] Starting: cycle={cycle}, unit_idx={unit_start}, "
            f"produced={produced}/{self.target_count()}"
        )

        while produced < self.target_count():
            for i in range(unit_start, len(units)):
                if produced >= self.target_count():
                    break
                unit = units[i]
                if fail_fast:
                    # See StageNode._run_one for rationale: no try/except so
                    # debuggers can break at the real failure site inside
                    # process_unit().
                    self.process_unit(unit, cycle)
                    self._save_checkpoint(cycle, i + 1)
                    self.progress.clear_failed(unit)
                    self._unit_fail_streak.pop(unit, None)
                    produced = len(self.progress.done_items())
                    log.debug(
                        f"[{self.name}] cycle={cycle} unit={unit!r} "
                        f"-> produced={produced}/{self.target_count()}"
                    )
                    continue
                try:
                    self.process_unit(unit, cycle)
                    self._save_checkpoint(cycle, i + 1)
                    self.progress.clear_failed(unit)
                    self._unit_fail_streak.pop(unit, None)
                    produced = len(self.progress.done_items())
                    log.debug(
                        f"[{self.name}] cycle={cycle} unit={unit!r} "
                        f"-> produced={produced}/{self.target_count()}"
                    )
                except Exception:
                    tb = traceback.format_exc()
                    log.warning(
                        f"[{self.name}] cycle={cycle} unit={unit!r} failed:\n{tb}"
                    )
                    # Record so `errors.db` / `progress.stats()['failed']` reflect
                    # the failure (unlike a per-item StageNode, the key here is
                    # the *unit*, not a produced item id).
                    self.progress.mark_failed(unit, tb)
                    streak = self._unit_fail_streak.get(unit, 0) + 1
                    self._unit_fail_streak[unit] = streak
                    if streak >= self._max_consecutive_unit_failures:
                        raise RuntimeError(
                            f"[{self.name}] unit {unit!r} failed "
                            f"{streak} times in a row (cycle={cycle}); aborting "
                            f"stage. Last error:\n{tb}"
                        ) from None
                    # Otherwise: skip this unit for now; it will be retried
                    # next cycle.

            unit_start = 0
            cycle += 1

        log.info(
            f"[{self.name}] Target reached: produced={produced}/{self.target_count()}"
        )

    # -------------------------------------------------------------------------
    # Checkpoint helpers
    # -------------------------------------------------------------------------

    def _load_checkpoint(self) -> tuple[int, int]:
        """Return ``(cycle, unit_idx)`` to resume from.  Defaults to ``(0, 0)``."""
        cycle_s = self._ckpt.get("cycle")
        unit_s = self._ckpt.get("unit_idx")
        if cycle_s is None or unit_s is None:
            return 0, 0
        return int(cycle_s), int(unit_s)

    def _save_checkpoint(self, cycle: int, unit_idx: int) -> None:
        """Record that ``(cycle, unit_idx - 1)`` has been successfully processed."""
        self._ckpt.set("cycle", str(cycle))
        self._ckpt.set("unit_idx", str(unit_idx))


class ParallelWhileSourceNode(WhileSourceNode):
    """
    A while-source node with parallel unit computation and ordered commits.

    Subclasses split unit handling into two phases:

    - ``compute_unit(unit, cycle)`` may run concurrently and must not write
      stage stores or progress.
    - ``commit_unit(unit, cycle, result)`` runs on the coordinator thread in
      the same deterministic unit order as ``WhileSourceNode``.

    This keeps the output item set stable while allowing expensive per-unit
    loads and transforms to overlap.
    """

    def __init__(
        self,
        name: str,
        store_dir: Path,
        deps: list[StageNode] | None = None,
        n_unit_workers: int = 1,
    ) -> None:
        super().__init__(name, store_dir, deps=deps)
        self._n_unit_workers = max(1, int(n_unit_workers))
        self.n_workers = self._n_unit_workers

    @abstractmethod
    def compute_unit(self, unit: str, cycle: int) -> Any:
        """Compute deterministic output for one ``(unit, cycle)`` pair."""
        ...

    @abstractmethod
    def commit_unit(self, unit: str, cycle: int, result: Any) -> None:
        """Commit a computed unit result. Called serially in unit order."""
        ...

    # --- Optional multi-GPU compute path -----------------------------------

    _gpu_batch_mult: int = 3  # in-flight units per GPU before an ordered commit

    def gpu_pool_spec(self):
        """Optional multi-GPU compute spec for the parallel-while loop.

        Return ``None`` (default) to use the thread/serial compute path, or a
        tuple ``(devices, init_fn, init_args, worker_fn)`` to dispatch
        ``compute_unit`` work across one spawned process per GPU. The
        coordinator still commits results in deterministic unit order, so the
        produced set and the exact target cutoff stay independent of worker
        scheduling.

        Worker contract:
          - ``init_fn(ctx, *init_args)`` builds per-worker state (becomes
            ``ctx.data``); JAX/GPU init must happen here, not at import time.
          - ``worker_fn(ctx, task)`` where ``task = (cycle, unit_idx, unit)``
            returns the same object ``compute_unit`` would, computed purely
            from ``(unit, cycle)`` (no worker/device-derived randomness).
        """
        return None

    def _run_gpu_pool(
        self,
        units: list[str],
        cycle: int,
        unit_start: int,
        produced: int,
        fail_fast: bool,
    ) -> int:
        """Drive compute across one process per GPU, committing in unit order.

        Determinism: workers compute in arbitrary completion order, but results
        are committed strictly in ascending ``(cycle, unit_idx)`` order and the
        target cutoff is applied within that ordered loop — identical semantics
        to the thread path, just distributed across GPUs.
        """
        from datasyn.utils.parallel.gpu_pool import GPUProcessPool

        spec = self.gpu_pool_spec()
        assert spec is not None
        devices, init_fn, init_args, worker_fn = spec
        batch = max(1, len(devices)) * max(1, self._gpu_batch_mult)
        target = self.target_count()

        log.info(f"[{self.name}] GPU pool: devices={devices} batch={batch}")
        pool = GPUProcessPool(devices, make_data=init_fn, init_args=init_args)
        try:
            while produced < target:
                # Next ordered batch of tasks, wrapping into the next cycle.
                tasks: list[tuple[int, int, str]] = []
                while len(tasks) < batch:
                    if unit_start >= len(units):
                        cycle += 1
                        unit_start = 0
                    tasks.append((cycle, unit_start, units[unit_start]))
                    unit_start += 1

                # Compute the batch across GPUs (completion order arbitrary).
                results: dict[tuple[int, int], Any] = {}
                for out in pool.imap_unordered(worker_fn, tasks, ()):
                    cy, ui, unit = out.idx
                    if out.err is not None:
                        log.warning(
                            f"[{self.name}] cycle={cy} unit={unit!r} failed:\n"
                            f"{out.err.tb}"
                        )
                        self.progress.mark_failed(unit, out.err.tb)
                        if fail_fast:
                            raise RuntimeError(
                                f"[{self.name}] {unit!r} failed:\n{out.err.tb}"
                            )
                    else:
                        results[(cy, ui)] = (unit, out.ret)

                # Commit in deterministic (cycle, unit_idx) order; cut off at N.
                for cy, ui, unit in tasks:
                    if produced >= target:
                        break
                    got = results.get((cy, ui))
                    if got is None:
                        continue  # failed unit: nothing to commit
                    _, result = got
                    self.commit_unit(unit, cy, result)
                    self._save_checkpoint(cy, ui + 1)
                    self.progress.clear_failed(unit)
                    produced = len(self.progress.done_items())
        finally:
            pool.close()
            pool.join()
        return produced

    def process_unit(self, unit: str, cycle: int) -> None:
        self.commit_unit(unit, cycle, self.compute_unit(unit, cycle))

    def run(
        self,
        fail_fast: bool = False,
    ) -> None:
        units = self.work_units()
        if not units:
            log.warning(f"[{self.name}] work_units() is empty; nothing to do.")
            return

        cycle, unit_start = self._load_checkpoint()
        produced = len(self.progress.done_items())

        log.info(
            f"[{self.name}] Starting: cycle={cycle}, unit_idx={unit_start}, "
            f"produced={produced}/{self.target_count()}, "
            f"unit_workers={self._n_unit_workers}"
        )

        if self.gpu_pool_spec() is not None:
            produced = self._run_gpu_pool(
                units,
                cycle,
                unit_start,
                produced,
                fail_fast=fail_fast,
            )
            log.info(
                f"[{self.name}] Target reached (gpu): "
                f"produced={produced}/{self.target_count()}"
            )
            return

        while produced < self.target_count():
            if self._n_unit_workers <= 1:
                produced = self._run_cycle_serial(
                    units,
                    cycle,
                    unit_start,
                    produced,
                    fail_fast=fail_fast,
                )
            else:
                produced = self._run_cycle_parallel(
                    units,
                    cycle,
                    unit_start,
                    produced,
                    fail_fast=fail_fast,
                )

            unit_start = 0
            cycle += 1

        log.info(
            f"[{self.name}] Target reached: produced={produced}/{self.target_count()}"
        )

    def _run_cycle_serial(
        self,
        units: list[str],
        cycle: int,
        unit_start: int,
        produced: int,
        fail_fast: bool,
    ) -> int:
        for i in range(unit_start, len(units)):
            if produced >= self.target_count():
                break
            unit = units[i]
            try:
                self.process_unit(unit, cycle)
                self._save_checkpoint(cycle, i + 1)
                self.progress.clear_failed(unit)
                self._unit_fail_streak.pop(unit, None)
                produced = len(self.progress.done_items())
                log.debug(
                    f"[{self.name}] cycle={cycle} unit={unit!r} "
                    f"-> produced={produced}/{self.target_count()}"
                )
            except Exception:
                if fail_fast:
                    raise
                tb = traceback.format_exc()
                log.warning(f"[{self.name}] cycle={cycle} unit={unit!r} failed:\n{tb}")
                self.progress.mark_failed(unit, tb)
                streak = self._unit_fail_streak.get(unit, 0) + 1
                self._unit_fail_streak[unit] = streak
                if streak >= self._max_consecutive_unit_failures:
                    raise RuntimeError(
                        f"[{self.name}] unit {unit!r} failed "
                        f"{streak} times in a row (cycle={cycle}); aborting "
                        f"stage. Last error:\n{tb}"
                    ) from None
        return produced

    def _run_cycle_parallel(
        self,
        units: list[str],
        cycle: int,
        unit_start: int,
        produced: int,
        fail_fast: bool,
    ) -> int:
        executor = ThreadPoolExecutor(max_workers=self._n_unit_workers)
        futures = {}
        next_i = unit_start
        cancel_pending = False

        def submit_until_full() -> None:
            nonlocal next_i
            while len(futures) < self._n_unit_workers and next_i < len(units):
                unit = units[next_i]
                futures[next_i] = executor.submit(self.compute_unit, unit, cycle)
                next_i += 1

        try:
            submit_until_full()
            for i in range(unit_start, len(units)):
                if produced >= self.target_count():
                    cancel_pending = True
                    break

                future = futures.pop(i)
                unit = units[i]
                try:
                    result = future.result()
                    self.commit_unit(unit, cycle, result)
                    self._save_checkpoint(cycle, i + 1)
                    self.progress.clear_failed(unit)
                    self._unit_fail_streak.pop(unit, None)
                    produced = len(self.progress.done_items())
                    log.debug(
                        f"[{self.name}] cycle={cycle} unit={unit!r} "
                        f"-> produced={produced}/{self.target_count()}"
                    )
                except Exception:
                    if fail_fast:
                        raise
                    tb = traceback.format_exc()
                    log.warning(
                        f"[{self.name}] cycle={cycle} unit={unit!r} failed:\n{tb}"
                    )
                    self.progress.mark_failed(unit, tb)
                    streak = self._unit_fail_streak.get(unit, 0) + 1
                    self._unit_fail_streak[unit] = streak
                    if streak >= self._max_consecutive_unit_failures:
                        cancel_pending = True
                        raise RuntimeError(
                            f"[{self.name}] unit {unit!r} failed "
                            f"{streak} times in a row (cycle={cycle}); aborting "
                            f"stage. Last error:\n{tb}"
                        ) from None

                if produced < self.target_count():
                    submit_until_full()
        finally:
            if cancel_pending:
                for future in futures.values():
                    future.cancel()
            executor.shutdown(wait=True, cancel_futures=cancel_pending)

        return produced
