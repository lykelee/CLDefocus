"""
Persistent per-item progress tracker for a pipeline stage.

Backed by WAL-mode SQLite — thread-safe and process-safe.
"""

from __future__ import annotations

from pathlib import Path

from datasyn.utils.files import ensure_directory
from datasyn.utils.sqlite.kvstore import KVStore
from datasyn.utils.sqlite.setstore import SetStore


class ItemProgress:
    """
    Tracks per-item state for a single stage.

    Three terminal states:
        done    — process() returned normally.
                  Item propagates to downstream stages.
        skipped — process() called mark_skipped().
                  Item does NOT propagate downstream (e.g., filtered out).
        failed  — process() raised an exception.
                  Item will be retried on the next run.

    Stored under a directory as three SQLite files:
        done.db     — SetStore
        skipped.db  — SetStore
        errors.db   — KVStore  (item_id -> latest traceback string)

    Thread-safe and process-safe (WAL-mode SQLite, one connection per op).
    """

    def __init__(self, dir: Path) -> None:
        dir = Path(dir)
        ensure_directory(dir)
        self._done = SetStore(dir / "done.db")
        self._skipped = SetStore(dir / "skipped.db")
        self._errors = KVStore(dir / "errors.db")

    # -------------------------------------------------------------------------
    # Marking
    # -------------------------------------------------------------------------

    def mark_done(self, item_id: str) -> None:
        """Record item as successfully processed. Clears any previous failure."""
        self._done.add(item_id)
        self._errors.delete(item_id)

    def mark_done_many(self, item_ids) -> None:
        """Batch-record many items as done in one transaction.

        Skips the per-item error clear (fresh items have no error entry); use
        ``mark_done`` for the retry-after-failure path.
        """
        ids = list(item_ids)
        if ids:
            self._done.update(ids)

    def mark_skipped(self, item_id: str) -> None:
        """
        Record item as intentionally excluded.
        Skipped items are not propagated to downstream stages.
        """
        self._skipped.add(item_id)

    def mark_failed(self, item_id: str, error: str) -> None:
        """Record item as failed with the given error string (e.g., a traceback)."""
        self._errors.set(item_id, error)

    def clear_failed(self, item_id: str) -> None:
        """Remove item_id from errors.db without marking it done or skipped."""
        self._errors.delete(item_id)

    def skip_all_failed(self) -> int:
        """
        Move all currently failed items to skipped state.

        Failed items will no longer be retried and will not propagate downstream.
        Returns the number of items moved.
        """
        failed_ids = list(self._errors.keys())
        for item_id in failed_ids:
            self._skipped.add(item_id)
            self._errors.delete(item_id)
        return len(failed_ids)

    # -------------------------------------------------------------------------
    # Querying
    # -------------------------------------------------------------------------

    def is_done(self, item_id: str) -> bool:
        return item_id in self._done

    def is_skipped(self, item_id: str) -> bool:
        return item_id in self._skipped

    def is_failed(self, item_id: str) -> bool:
        return self._errors.contains(item_id)

    def done_items(self) -> frozenset[str]:
        return frozenset(self._done)

    def skipped_items(self) -> frozenset[str]:
        return frozenset(self._skipped)

    def failed_items(self) -> dict[str, str]:
        """Return {item_id: error_message} for all currently failed items."""
        return dict(self._errors.items())

    def processable(self) -> frozenset[str]:
        """
        Items that are done or skipped — will not be processed again.

        Used internally by StageNode to determine what has already been handled.
        Note: only `done` items propagate downstream; `skipped` items stop here.
        """
        return self.done_items() | self.skipped_items()

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        return {
            "done": len(self._done),
            "skipped": len(self._skipped),
            "failed": len(self._errors),
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"ItemProgress("
            f"done={s['done']}, skipped={s['skipped']}, failed={s['failed']})"
        )
