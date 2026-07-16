"""
SharpnessStore: stores per-patch raw sharpness components for later filtering.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from datasyn.synthesis.common import IOMode
from datasyn.utils.files import ensure_directory
from datasyn.utils.sqlite.tablestore import SQLiteTableStore

DEFAULT_SHARPNESS_KERNEL_SIZES: tuple[int, ...] = (3, 7, 11, 15)

_SCORE_COLUMNS = tuple(f"score_k{ks:02d}" for ks in DEFAULT_SHARPNESS_KERNEL_SIZES)
_TABLE_COLUMNS: dict[str, type] = {"name": str} | {col: float for col in _SCORE_COLUMNS}


class SharpnessStore:
    """SQLite-backed store for raw multi-scale sharpness scores."""

    def __init__(self, path: Path, mode: IOMode = IOMode.READ) -> None:
        path = Path(path)
        readonly = mode == IOMode.READ
        if not readonly:
            ensure_directory(path.parent)
        self._table = SQLiteTableStore(
            path,
            _TABLE_COLUMNS,
            primary_key="name",
            readonly=readonly,
        )

    def add(self, name: str, scores: np.ndarray) -> None:
        scores = np.asarray(scores, dtype=np.float32)
        if scores.shape != (len(_SCORE_COLUMNS),):
            raise ValueError(
                f"SharpnessStore expects shape {(len(_SCORE_COLUMNS),)}, got {scores.shape}"
            )
        if self._table.get("name", name):
            return
        row = {"name": name}
        row.update({col: float(scores[i]) for i, col in enumerate(_SCORE_COLUMNS)})
        self._table.insert(row)

    def add_many(self, names, scores_list) -> None:
        """Batch-insert many (name, scores) rows in one transaction."""
        existing = set(self._table.get_column("name"))
        rows = []
        for name, scores in zip(names, scores_list):
            if name in existing:
                continue
            scores = np.asarray(scores, dtype=np.float32)
            row = {"name": name}
            row.update(
                {col: float(scores[i]) for i, col in enumerate(_SCORE_COLUMNS)}
            )
            rows.append(row)
        if rows:
            self._table.insert_many(rows)

    def load(self, name: str) -> np.ndarray:
        rows = self._table.get("name", name)
        if not rows:
            raise KeyError(f"SharpnessStore: unknown patch '{name}'")
        row = rows[0]
        return np.asarray([row[col] for col in _SCORE_COLUMNS], dtype=np.float32)

    def load_many(self, names: list[str]) -> np.ndarray:
        rows = self._table.get_all()
        score_map = {
            row["name"]: np.asarray(
                [row[col] for col in _SCORE_COLUMNS], dtype=np.float32
            )
            for row in rows
        }
        missing = [name for name in names if name not in score_map]
        if missing:
            raise KeyError(
                "SharpnessStore missing score rows for patch ids: "
                + ", ".join(missing[:5])
                + ("..." if len(missing) > 5 else "")
            )
        return np.stack([score_map[name] for name in names], axis=0)

    def get_all_names(self) -> list[str]:
        return sorted(self._table.get_column("name"))
