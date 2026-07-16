"""
Pipeline: DAG container and orchestrator.
"""

from __future__ import annotations

import logging
from collections import Counter, deque

from .node import StageNode

log = logging.getLogger(__name__)


class Pipeline:
    """
    A data-parallel DAG pipeline over a set of StageNodes.

    Validates the DAG (no cycles, all deps registered) on construction
    and computes a topological order used for execution.

    Execution via ``run_batch()``: topological order, one stage at a time.
    Each stage fully completes before the next one starts.

    Example::

        pipe = Pipeline([source, stage_a, stage_b])
        pipe.run_batch()
        print(pipe.stats())
    """

    def __init__(self, nodes: list[StageNode]) -> None:
        if not nodes:
            raise ValueError("Pipeline must have at least one node.")

        # Validate unique names.
        names = [n.name for n in nodes]
        if len(names) != len(set(names)):
            dups = [name for name, c in Counter(names).items() if c > 1]
            raise ValueError(f"Duplicate node names: {dups}")

        # Validate all deps are registered in this pipeline.
        node_set = {n.name for n in nodes}
        for node in nodes:
            for dep in node.deps:
                if dep.name not in node_set:
                    raise ValueError(
                        f"Node '{node.name}' depends on '{dep.name}', "
                        f"which is not registered in this pipeline."
                    )

        self._nodes = list(nodes)
        self._topo_order = _topological_sort(nodes)

    @property
    def nodes(self) -> list[StageNode]:
        return list(self._nodes)

    def run_batch(
        self,
        fail_fast: bool = False,
        show_progress: bool = True,
    ) -> None:
        """
        Run all stages in topological order, one at a time.
        Each stage fully completes before the next one starts.

        Args:
            show_progress: Show live tqdm progress bars (one per stage).
        """
        from .monitor import PipelineProgressMonitor

        with PipelineProgressMonitor(self._topo_order) if show_progress else _nullctx():
            for node in self._topo_order:
                log.info(f"[pipeline] Starting: {node.name}")
                node.run(fail_fast=fail_fast)
                log.info(f"[pipeline] Finished: {node.name}  {node.progress}")

    def stats(self) -> dict[str, dict[str, int]]:
        """Return a progress summary {stage_name: {done, skipped, failed}} for all stages."""
        return {node.name: node.progress.stats() for node in self._topo_order}

    def __repr__(self) -> str:
        order = " -> ".join(n.name for n in self._topo_order)
        return f"Pipeline([{order}])"


class _nullctx:
    """No-op context manager used when show_progress=False."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def _topological_sort(nodes: list[StageNode]) -> list[StageNode]:
    """
    Kahn's algorithm. Returns nodes in topological order.
    Raises ValueError if the graph contains a cycle.
    """
    in_degree: dict[str, int] = {n.name: 0 for n in nodes}
    name2node: dict[str, StageNode] = {n.name: n for n in nodes}

    # dependents[x] = list of node names that directly depend on x
    dependents: dict[str, list[str]] = {n.name: [] for n in nodes}
    for node in nodes:
        for dep in node.deps:
            in_degree[node.name] += 1
            dependents[dep.name].append(node.name)

    queue = deque(n for n in nodes if in_degree[n.name] == 0)
    order: list[StageNode] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for child_name in dependents[node.name]:
            in_degree[child_name] -= 1
            if in_degree[child_name] == 0:
                queue.append(name2node[child_name])

    if len(order) != len(nodes):
        processed = {n.name for n in order}
        cycle_nodes = [n.name for n in nodes if n.name not in processed]
        raise ValueError(
            f"Cycle detected in pipeline DAG involving nodes: {cycle_nodes}"
        )

    return order
