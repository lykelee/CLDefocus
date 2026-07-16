"""
Persistent set backed by SQLite.

Thread-safe and process-safe for multiple readers/writers.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path
from typing import Iterable, Iterator


class SetStore:
    """
    A persistent set backed by SQLite.

    Features:
    - Thread-safe and process-safe (uses WAL mode)
    - Multiple concurrent readers, serialized writers
    - Full set algebra (union, intersection, difference, etc.)

    Usage:
        s = SQLiteSet("my_set.db")
        s.add("a")
        s.add("b")
        print("a" in s)  # True
        print(len(s))     # 2

    Read-only mode:
        s = SQLiteSet("my_set.db", readonly=True)
        print("a" in s)  # OK
        s.add("x")       # Raises PermissionError

    Or with context manager for explicit cleanup:
        with SQLiteSet("my_set.db") as s:
            s.add("a")
    """

    def __init__(self, path: str | Path, *, readonly: bool = False):
        """
        Initialize the set store.

        Args:
            path: Path to the SQLite database file. Created if it doesn't exist
                  (unless readonly=True).
            readonly: If True, open in read-only mode. The file must exist.
                      Write operations will raise PermissionError.
        """
        self.path = Path(path)
        self.readonly = readonly

        if readonly:
            if not self.path.exists():
                raise FileNotFoundError(f"Database file not found: {self.path}")
        else:
            self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS items (value TEXT PRIMARY KEY)")
            conn.execute("PRAGMA journal_mode=WAL")

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """
        Open a connection, commit (or rollback) on exit, then close it.

        See ``SQLiteTableStore._connect`` for the rationale; keeping the
        handle open past the ``with`` block was the pre-fix behaviour and
        caused WAL sidecars to linger after hard process kills.

        Uses timeout=30s to wait for locks rather than failing immediately.
        In readonly mode, uses SQLite URI mode to enforce read-only access.
        """
        if self.readonly:
            uri = f"file:{self.path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=30.0)
        else:
            conn = sqlite3.connect(self.path, timeout=30.0)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _check_writable(self) -> None:
        """Raise PermissionError if store is read-only."""
        if self.readonly:
            raise PermissionError("SQLiteSet is opened in read-only mode")

    # -------------------------------------------------------------------------
    # Core operations
    # -------------------------------------------------------------------------

    def add(self, value: str) -> None:
        """Add a value to the set. No effect if already present."""
        self._check_writable()
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO items (value) VALUES (?)", (value,))
            conn.commit()

    def discard(self, value: str) -> None:
        """Remove a value from the set. No error if missing."""
        self._check_writable()
        with self._connect() as conn:
            conn.execute("DELETE FROM items WHERE value = ?", (value,))
            conn.commit()

    def remove(self, value: str) -> None:
        """Remove a value from the set. Raises KeyError if missing."""
        self._check_writable()
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM items WHERE value = ?", (value,))
            conn.commit()
            if cursor.rowcount == 0:
                raise KeyError(value)

    def clear(self) -> int:
        """Delete all entries. Returns number of deleted entries."""
        self._check_writable()
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM items")
            conn.commit()
            return cursor.rowcount

    def __contains__(self, value: object) -> bool:
        """Check if value exists in the set."""
        if not isinstance(value, str):
            return False
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM items WHERE value = ?", (value,)
            ).fetchone()
            return row is not None

    def __len__(self) -> int:
        """Return number of entries."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM items").fetchone()
            return row[0]

    def __iter__(self) -> Iterator[str]:
        """Iterate over values."""
        with self._connect() as conn:
            rows = conn.execute("SELECT value FROM items ORDER BY value").fetchall()
            return iter(row[0] for row in rows)

    # -------------------------------------------------------------------------
    # Bulk operations
    # -------------------------------------------------------------------------

    def update(self, *others: Iterable[str]) -> None:
        """Add all items from one or more iterables."""
        self._check_writable()
        with self._connect() as conn:
            for other in others:
                conn.executemany(
                    "INSERT OR IGNORE INTO items (value) VALUES (?)",
                    ((v,) for v in other),
                )
            conn.commit()

    def to_list(self) -> list[str]:
        """Return all values as a sorted list."""
        with self._connect() as conn:
            rows = conn.execute("SELECT value FROM items ORDER BY value").fetchall()
            return [row[0] for row in rows]

    def to_set(self) -> set[str]:
        """Return all values as a Python set."""
        with self._connect() as conn:
            rows = conn.execute("SELECT value FROM items").fetchall()
            return {row[0] for row in rows}

    # -------------------------------------------------------------------------
    # Set algebra (return plain Python sets)
    # -------------------------------------------------------------------------

    def union(self, other: Iterable[str]) -> set[str]:
        """Return the union of this set and another iterable."""
        return self.to_set() | set(other)

    def intersection(self, other: Iterable[str]) -> set[str]:
        """Return the intersection of this set and another iterable."""
        return self.to_set() & set(other)

    def difference(self, other: Iterable[str]) -> set[str]:
        """Return the difference of this set and another iterable."""
        return self.to_set() - set(other)

    def symmetric_difference(self, other: Iterable[str]) -> set[str]:
        """Return the symmetric difference of this set and another iterable."""
        return self.to_set() ^ set(other)

    def issubset(self, other: Iterable[str]) -> bool:
        """Return True if this set is a subset of another iterable."""
        return self.to_set() <= set(other)

    def issuperset(self, other: Iterable[str]) -> bool:
        """Return True if this set is a superset of another iterable."""
        return self.to_set() >= set(other)

    def isdisjoint(self, other: Iterable[str]) -> bool:
        """Return True if this set has no elements in common with another."""
        return self.to_set().isdisjoint(set(other))

    def __or__(self, other: Iterable[str]) -> set[str]:
        return self.union(other)

    def __and__(self, other: Iterable[str]) -> set[str]:
        return self.intersection(other)

    def __sub__(self, other: Iterable[str]) -> set[str]:
        return self.difference(other)

    def __xor__(self, other: Iterable[str]) -> set[str]:
        return self.symmetric_difference(other)

    def __le__(self, other: Iterable[str]) -> bool:
        return self.issubset(other)

    def __ge__(self, other: Iterable[str]) -> bool:
        return self.issuperset(other)

    # -------------------------------------------------------------------------
    # Context manager
    # -------------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
        return None

    def close(self) -> None:
        """
        Explicitly close the store and checkpoint the WAL.

        This is optional - connections are closed after each operation.
        However, calling close() forces a WAL checkpoint which can be
        useful before copying the database file.

        In read-only mode, this is a no-op (no checkpoint needed).
        """
        if self.readonly:
            return
        with self._connect() as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    # -------------------------------------------------------------------------
    # Import/export
    # -------------------------------------------------------------------------

    def to_json(self, path: str | Path, indent: int | None = 2) -> None:
        """Export all values to a JSON file as an array."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_list(), f, indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, db_path: str | Path, json_path: str | Path) -> SetStore:
        """
        Create a SQLiteSet from a JSON file.

        Expects format: ["value1", "value2", ...]
        """
        store = cls(db_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        store.update(data)
        return store

    # -------------------------------------------------------------------------
    # Repr
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        mode = "readonly" if self.readonly else "rw"
        return f"SQLiteSet({self.path!r}, {mode}, len={len(self)})"
