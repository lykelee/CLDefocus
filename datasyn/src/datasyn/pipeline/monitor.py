"""
PipelineProgressMonitor — live tqdm progress bars for Pipeline runs.

One bar per stage, updated by a single background thread that polls
``node.progress.stats()`` periodically.  Only the monitor thread ever
touches the bars, so there are no inter-thread rendering races.

Each bar's total is the per-stage expected item count:

- Root node: ``progress_total()`` if defined (e.g., ``WhileSourceNode``,
  ``SubprocessStageRunNode``), otherwise ``len(source_items())``.
- Non-root node: the minimum of its deps' totals (items can only flow
  through once every upstream dep produces them).

For independent root stages over the same item set, each stage's bar now
shows progress against its own total (not the pipeline-wide sum).
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .node import StageNode


class PipelineProgressMonitor:
    """
    Context manager that renders one ``tqdm`` bar per stage.

    Args:
        nodes:          Stage nodes to monitor, in topological order.
        poll_interval:  Seconds between stats polls.  0.5 s is a good default;
                        go higher (2-5 s) for very fast stages to reduce flicker.

    Usage::

        with PipelineProgressMonitor(pipe.nodes):
            pipe.run_batch()
    """

    def __init__(
        self,
        nodes: list[StageNode],
        poll_interval: float = 0.5,
    ) -> None:
        self._nodes = nodes
        self._poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._bars: list = []
        self._stage_totals: dict[str, int | None] = _compute_stage_totals(nodes)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "PipelineProgressMonitor":
        try:
            from tqdm.auto import tqdm
        except ImportError:
            return self  # silently skip if tqdm is not installed

        # Pre-fill bars with current state so resume runs look right.
        self._bars = []
        for i, node in enumerate(self._nodes):
            stats = node.progress.stats()
            processed = stats["done"] + stats["skipped"]
            bar = tqdm(
                total=self._stage_totals.get(node.name),
                initial=processed,
                position=i,
                desc=f"{node.name:<26}",
                unit="item",
                leave=True,
                dynamic_ncols=True,
                colour="green",
            )
            bar.set_postfix(
                done=stats["done"],
                skip=stats["skipped"],
                fail=stats["failed"],
                refresh=False,
            )
            self._bars.append(bar)

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="pipeline-monitor"
        )
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        # Final update so bars end at their true value.
        for bar, node in zip(self._bars, self._nodes):
            self._update_bar(bar, node)
            bar.close()
        # Blank line after the last bar.
        try:
            from tqdm.auto import tqdm

            tqdm.write("")
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        while not self._stop.wait(self._poll_interval):
            for bar, node in zip(self._bars, self._nodes):
                self._update_bar(bar, node)

    @staticmethod
    def _update_bar(bar, node: "StageNode") -> None:
        stats = node.progress.stats()
        bar.n = stats["done"] + stats["skipped"]

        # Dynamic total: if the node publishes ``progress_total()`` and it now
        # reports a real number (e.g., a node that sub-samples from upstream
        # only knows its own total once deps finish), supersede the static
        # fallback chosen at startup. Nothing to update if the node doesn't
        # define the method or it still returns None.
        new_total = _effective_total(node)
        if new_total is not None and new_total != bar.total:
            bar.total = new_total

        bar.set_postfix(
            done=stats["done"],
            skip=stats["skipped"],
            fail=stats["failed"],
            refresh=False,
        )
        bar.refresh()


def _compute_stage_totals(
    nodes: list["StageNode"],
) -> dict[str, int | None]:
    """
    Per-stage expected item count.

    Resolution order for each node (first non-None wins):

    1. Node's own ``progress_total()`` if defined and returning a value.
       This applies regardless of whether the node has deps — nodes that
       sub-sample or otherwise don't emit one item per upstream item need
       to speak for themselves.
    2. Root nodes (no deps): ``len(source_items())``.
    3. Non-root nodes: ``min(dep_totals)`` — items can only flow through
       once every upstream has produced them, so the effective total is
       bounded by the slowest upstream.

    ``progress_total()`` may return None at startup (e.g., a sub-sampling
    node that waits for deps to finish before drawing its sample); in that
    case this falls back to steps 2/3 and ``_update_bar`` re-reads
    ``progress_total()`` on each poll to pick up the real number later.
    """
    totals: dict[str, int | None] = {}
    for node in nodes:
        totals[node.name] = _effective_total(node, memo=totals)
    return totals


def _effective_total(
    node: "StageNode",
    memo: dict[str, int | None] | None = None,
) -> int | None:
    """Current best-known total for a node, following dependency totals."""
    if memo is None:
        memo = {}
    if node.name in memo:
        return memo[node.name]

    own = _node_own_total(node)
    if own is not None:
        memo[node.name] = own
        return own

    if not node.deps:
        total = _source_items_total(node)
        memo[node.name] = total
        return total

    dep_totals = [_effective_total(dep, memo) for dep in node.deps]
    if any(t is None for t in dep_totals):
        memo[node.name] = None
        return None

    total = min(dep_totals)  # type: ignore[type-var]
    memo[node.name] = total
    return total


def _node_own_total(node: "StageNode") -> int | None:
    """
    Node's explicitly-published total via ``progress_total()``.

    Returns None if the node doesn't override the method, or does but
    currently signals "unknown" (e.g., waiting for deps).
    """
    if hasattr(node, "progress_total"):
        t = node.progress_total()
        if t is not None:
            return int(t)
    return None


def _source_items_total(node: "StageNode") -> int | None:
    """Fallback total via ``len(source_items())``; None if that fails."""
    try:
        return len(node.source_items())
    except Exception:
        return None
