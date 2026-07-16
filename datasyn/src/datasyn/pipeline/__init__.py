"""
General persistent data-parallel DAG pipeline framework.

Key abstractions:

    ItemProgress
        Tracks per-item state (done / skipped / failed) for a single stage,
        backed by WAL-mode SQLite. Thread-safe and process-safe.

    StageNode (ABC)
        One stage in the pipeline. Subclass and implement ``process(item_id)``.
        Root nodes (no deps) also implement ``source_items()``.

    Pipeline
        Holds a set of StageNodes forming a DAG. Handles validation (no cycles),
        topological ordering, and sequential execution via ``run_batch()``
        (one stage at a time).
"""

from .monitor import PipelineProgressMonitor
from .node import GroupedStageNode, ParallelWhileSourceNode, StageNode, WhileSourceNode
from .pipeline import Pipeline
from .progress import ItemProgress

__all__ = [
    "ItemProgress",
    "StageNode",
    "GroupedStageNode",
    "WhileSourceNode",
    "ParallelWhileSourceNode",
    "Pipeline",
    "PipelineProgressMonitor",
]
