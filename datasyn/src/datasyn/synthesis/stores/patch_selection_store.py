"""
PatchSelectionStore: stores the selected patch ids produced by a filter stage.
"""

from __future__ import annotations

from pathlib import Path

from datasyn.synthesis.common import IOMode
from datasyn.utils.files import ensure_directory
from datasyn.utils.sqlite.setstore import SetStore


class PatchSelectionStore:
    """Persistent set of selected patch ids."""

    def __init__(self, path: Path, mode: IOMode = IOMode.READ) -> None:
        path = Path(path)
        readonly = mode == IOMode.READ
        if not readonly:
            ensure_directory(path.parent)
        self._set = SetStore(path, readonly=readonly)

    def add(self, item_id: str) -> None:
        self._set.add(item_id)

    def update(self, item_ids) -> None:
        self._set.update(item_ids)

    def clear(self) -> int:
        return self._set.clear()

    def list_names(self) -> list[str]:
        return self._set.to_list()

    def __contains__(self, item_id: object) -> bool:
        return item_id in self._set

    def __len__(self) -> int:
        return len(self._set)
