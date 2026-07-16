"""
Simple key-value store backed by SQLite.

Thread-safe and process-safe for multiple readers/writers.
"""

from __future__ import annotations

import contextlib
import csv
import json
import sqlite3
from pathlib import Path
from typing import Iterator


class KVStore:
    """
    A simple key-value store backed by SQLite.

    Features:
    - Thread-safe and process-safe (uses WAL mode)
    - Multiple concurrent readers, serialized writers
    - Human-readable inspection via to_csv(), to_json(), items()

    Usage:
        store = KVStore("my_store.db")
        store["key"] = "value"
        print(store["key"])
        del store["key"]

    Read-only mode:
        store = KVStore("my_store.db", readonly=True)
        print(store["key"])  # OK
        store["key"] = "x"   # Raises PermissionError

    Or with context manager for explicit cleanup:
        with KVStore("my_store.db") as store:
            store["key"] = "value"

    Note on string safety:
    - All strings are safe: whitespace, unicode, quotes, backslashes, newlines
    - Empty strings are valid (distinct from None/missing)
    - Only restriction: keys cannot be None
    """

    def __init__(self, path: str | Path, *, readonly: bool = False):
        """
        Initialize the KV store.

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
            conn.execute(
                "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            # WAL mode allows concurrent reads while writing
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
            # URI mode with ?mode=ro enforces read-only at SQLite level
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
            raise PermissionError("KVStore is opened in read-only mode")

    # -------------------------------------------------------------------------
    # Core operations
    # -------------------------------------------------------------------------

    def get(self, key: str, default: str | None = None) -> str | None:
        """Get value by key, returning default if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM kv WHERE key = ?", (key,)
            ).fetchone()
            return row[0] if row else default

    def set(self, key: str, value: str) -> None:
        """Set a key-value pair. Overwrites if key exists."""
        self._check_writable()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
                (key, value),
            )
            conn.commit()

    def delete(self, key: str) -> bool:
        """
        Delete a key. Returns True if key existed, False otherwise.
        """
        self._check_writable()
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM kv WHERE key = ?", (key,))
            conn.commit()
            return cursor.rowcount > 0

    def contains(self, key: str) -> bool:
        """Check if key exists."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM kv WHERE key = ?", (key,)
            ).fetchone()
            return row is not None

    def clear(self) -> int:
        """Delete all entries. Returns number of deleted entries."""
        self._check_writable()
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM kv")
            conn.commit()
            return cursor.rowcount

    # -------------------------------------------------------------------------
    # Iteration and bulk access
    # -------------------------------------------------------------------------

    def keys(self) -> list[str]:
        """Return all keys."""
        with self._connect() as conn:
            rows = conn.execute("SELECT key FROM kv ORDER BY key").fetchall()
            return [row[0] for row in rows]

    def values(self) -> list[str]:
        """Return all values."""
        with self._connect() as conn:
            rows = conn.execute("SELECT value FROM kv ORDER BY key").fetchall()
            return [row[0] for row in rows]

    def items(self) -> list[tuple[str, str]]:
        """Return all (key, value) pairs."""
        with self._connect() as conn:
            return conn.execute("SELECT key, value FROM kv ORDER BY key").fetchall()

    def __len__(self) -> int:
        """Return number of entries."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM kv").fetchone()
            return row[0]

    def __iter__(self) -> Iterator[str]:
        """Iterate over keys."""
        return iter(self.keys())

    # -------------------------------------------------------------------------
    # Dict-like interface
    # -------------------------------------------------------------------------

    def __getitem__(self, key: str) -> str:
        """Get value by key. Raises KeyError if not found."""
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: str) -> None:
        """Set a key-value pair."""
        self.set(key, value)

    def __delitem__(self, key: str) -> None:
        """Delete a key. Raises KeyError if not found."""
        if not self.delete(key):
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return self.contains(key)

    # -------------------------------------------------------------------------
    # Context manager
    # -------------------------------------------------------------------------

    def __enter__(self) -> KVStore:
        """
        Enter context manager.

        The context manager is useful for:
        - Explicit resource cleanup in long-running applications
        - Ensuring WAL checkpoint on exit
        - Signaling intent that the store usage is scoped

        Note: The store is still usable after exiting the context, but
        close() will have been called.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and close the store."""
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
    # Human-readable export
    # -------------------------------------------------------------------------

    def to_csv(self, path: str | Path) -> None:
        """
        Export all entries to a CSV file.

        Format: key,value (with header row)
        Handles embedded commas, quotes, and newlines correctly.
        """
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            writer.writerows(self.items())

    def to_json(self, path: str | Path, indent: int | None = 2) -> None:
        """
        Export all entries to a JSON file.

        Format: {"key1": "value1", "key2": "value2", ...}
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(self.items()), f, indent=indent, ensure_ascii=False)

    def to_dict(self) -> dict[str, str]:
        """Return all entries as a dict."""
        return dict(self.items())

    @classmethod
    def from_csv(cls, db_path: str | Path, csv_path: str | Path) -> KVStore:
        """
        Create a KVStore from a CSV file.

        Expects format: key,value (with or without header row)
        If first row is exactly "key,value", it's treated as header.
        """
        store = cls(db_path)
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            first_row = next(reader, None)
            if first_row is None:
                return store
            # Check if it's a header
            if first_row != ["key", "value"]:
                store.set(first_row[0], first_row[1])
            for row in reader:
                if len(row) >= 2:
                    store.set(row[0], row[1])
        return store

    @classmethod
    def from_json(cls, db_path: str | Path, json_path: str | Path) -> KVStore:
        """
        Create a KVStore from a JSON file.

        Expects format: {"key1": "value1", "key2": "value2", ...}
        """
        store = cls(db_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, value in data.items():
            store.set(key, value)
        return store

    # -------------------------------------------------------------------------
    # Repr
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        mode = "readonly" if self.readonly else "rw"
        return f"KVStore({self.path!r}, {mode}, len={len(self)})"
