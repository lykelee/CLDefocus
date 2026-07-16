"""
Persistent table store with abstract interface and SQLite backend.

Thread-safe and process-safe for multiple readers/writers.
"""

from __future__ import annotations

import contextlib
import csv
import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator

# -------------------------------------------------------------------------
# Abstract base class
# -------------------------------------------------------------------------


class TableStore(ABC):
    """
    Abstract base class for table storage backends.

    Defines a simple CRUD interface for tabular data. Filtering uses
    Django-style field lookups (see find()). No joins, no aggregations,
    no OR conditions — by design.
    """

    @property
    @abstractmethod
    def columns(self) -> dict[str, type]:
        """Return column definitions as {name: type}."""
        ...

    @abstractmethod
    def insert(self, record: dict[str, Any]) -> None:
        """Insert a single record. All columns must be present."""
        ...

    @abstractmethod
    def insert_many(self, records: list[dict[str, Any]]) -> None:
        """Insert multiple records in a single transaction."""
        ...

    @abstractmethod
    def get(self, column: str, value: Any) -> list[dict[str, Any]]:
        """Get all records where column equals value."""
        ...

    @abstractmethod
    def get_all(self) -> list[dict[str, Any]]:
        """Get all records."""
        ...

    @abstractmethod
    def find(self, **kwargs: Any) -> list[dict[str, Any]]:
        """
        Find records matching all given conditions (AND).

        Supports Django-style field lookups using __ (double underscore):
          - No suffix -> equality:          name="Alice"
          - __gt      -> greater than:       score__gt=90.0
          - __lt      -> less than:          score__lt=90.0
          - __gte     -> greater or equal:   score__gte=80.0
          - __lte     -> less or equal:      score__lte=95.0
          - __ne      -> not equal:          active__ne=True

        All conditions are combined with AND. Call get_all() for no filter.

        Examples:
            t.find(active=True)
            t.find(active=True, name="Alice")
            t.find(score__gt=90.0, active=True)
            t.find(score__gte=80.0, score__lte=95.0)
        """
        ...

    @abstractmethod
    def get_column(self, column: str, **kwargs: Any) -> list[Any]:
        """Get all values for a single column, with optional filtering."""
        ...

    @abstractmethod
    def update(
        self, where_column: str, where_value: Any, updates: dict[str, Any]
    ) -> int:
        """Update records where column equals value. Returns count updated."""
        ...

    @abstractmethod
    def delete(self, where_column: str, where_value: Any) -> int:
        """Delete records where column equals value. Returns count deleted."""
        ...

    @abstractmethod
    def clear(self) -> int:
        """Delete all records. Returns count deleted."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return number of records."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all records."""
        ...

    @abstractmethod
    def to_csv(self, path: str | Path) -> None:
        """Export all records to CSV."""
        ...

    @abstractmethod
    def to_json(self, path: str | Path, indent: int | None = 2) -> None:
        """Export all records to JSON."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources."""
        ...

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
        return None


# -------------------------------------------------------------------------
# SQLite implementation
# -------------------------------------------------------------------------

_TYPE_MAP: dict[type, str] = {
    str: "TEXT",
    int: "INTEGER",
    float: "REAL",
    bool: "INTEGER",
}

# Django-style field lookup suffixes -> SQL operators.
# Usage: score__gt=90  ->  WHERE score > 90
_LOOKUP_OPS: dict[str, str] = {
    "gt": ">",
    "lt": "<",
    "gte": ">=",
    "lte": "<=",
    "ne": "!=",
}


class SQLiteTableStore(TableStore):
    """
    A simple table store backed by SQLite.

    Features:
    - Thread-safe and process-safe (uses WAL mode)
    - CRUD with Django-style field lookup filtering (find())
    - Runtime type validation on insert and update
    - CSV/JSON import/export

    Usage:
        cols = {"id": int, "name": str, "score": float, "active": bool}
        store = SQLiteTableStore("data.db", cols, primary_key="id")
        store.insert({"id": 1, "name": "Alice", "score": 95.5, "active": True})
        store.find(active=True)                   # equality
        store.find(score__gt=90.0, active=True)   # comparison + AND

    Read-only mode:
        store = SQLiteTableStore("data.db", cols, readonly=True)
        store.get_all()      # OK
        store.insert({...})  # Raises PermissionError
    """

    def __init__(
        self,
        path: str | Path,
        columns: dict[str, type],
        *,
        table_name: str = "data",
        primary_key: str | None = None,
        readonly: bool = False,
    ):
        """
        Initialize the table store.

        Args:
            path: Path to the SQLite database file.
            columns: Dict mapping column names to Python types
                     (str, int, float, bool).
            table_name: Name of the SQL table.
            primary_key: Optional column name to use as primary key.
            readonly: If True, open in read-only mode.

        Raises:
            ValueError: If columns are empty, contain unsupported types,
                        or primary_key is not in columns.
            FileNotFoundError: If readonly=True and file doesn't exist.
        """
        self._columns = dict(columns)
        self._col_names = list(columns.keys())
        self._table_name = table_name
        self._primary_key = primary_key
        self.path = Path(path)
        self.readonly = readonly

        self._validate_columns()

        if readonly:
            if not self.path.exists():
                raise FileNotFoundError(f"Database file not found: {self.path}")
        else:
            self._init_db()

    def _validate_columns(self) -> None:
        """Validate column definitions."""
        if not self._columns:
            raise ValueError("columns must not be empty")
        for name, typ in self._columns.items():
            if typ not in _TYPE_MAP:
                raise ValueError(
                    f"Unsupported type {typ!r} for column {name!r}. "
                    f"Supported: {set(_TYPE_MAP.keys())}"
                )
        if self._primary_key is not None and self._primary_key not in self._columns:
            raise ValueError(f"primary_key {self._primary_key!r} is not in columns")

    def _init_db(self) -> None:
        """Initialize the database schema."""
        col_defs = []
        for name, typ in self._columns.items():
            sql_type = _TYPE_MAP[typ]
            suffix = " PRIMARY KEY" if name == self._primary_key else ""
            col_defs.append(f"{name} {sql_type}{suffix}")
        col_sql = ", ".join(col_defs)
        with self._connect() as conn:
            conn.execute(f"CREATE TABLE IF NOT EXISTS {self._table_name} ({col_sql})")
            conn.execute("PRAGMA journal_mode=WAL")

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """
        Open a connection, commit (or rollback) on exit, then close it.

        Used as ``with self._connect() as conn``. The enclosing ``with conn``
        inside this function gives sqlite3's usual commit-on-success /
        rollback-on-exception behaviour; the outer ``try/finally`` closes the
        underlying file handle so WAL sidecars (``*-wal``, ``*-shm``) get
        a chance to checkpoint and lock bytes are released. Without that
        close every operation would leak a connection, and processes killed
        mid-run would leave stale WAL state that some SQLite clients refuse
        to open.

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
            raise PermissionError("SQLiteTableStore is opened in read-only mode")

    def _validate_column_name(self, column: str) -> None:
        """Validate that a column name exists."""
        if column not in self._columns:
            raise KeyError(f"Unknown column: {column!r}")

    def _validate_record(self, record: dict[str, Any]) -> None:
        """Validate that record has all columns with correct types."""
        if set(record.keys()) != set(self._col_names):
            missing = set(self._col_names) - set(record.keys())
            extra = set(record.keys()) - set(self._col_names)
            parts = []
            if missing:
                parts.append(f"missing={missing}")
            if extra:
                parts.append(f"extra={extra}")
            raise ValueError(f"Record column mismatch: {', '.join(parts)}")
        for name, typ in self._columns.items():
            val = record[name]
            if not isinstance(val, typ):
                # Allow int where float is expected
                if typ is float and isinstance(val, int):
                    continue
                raise TypeError(
                    f"Column {name!r} expects {typ.__name__}, got {type(val).__name__}"
                )

    def _serialize(self, record: dict[str, Any]) -> tuple:
        """Serialize a record to a tuple of SQLite-compatible values."""
        values = []
        for name in self._col_names:
            val = record[name]
            if self._columns[name] is bool:
                val = int(val)
            values.append(val)
        return tuple(values)

    def _deserialize_row(self, row: tuple) -> dict[str, Any]:
        """Deserialize a SQLite row tuple to a dict."""
        result = {}
        for i, name in enumerate(self._col_names):
            val = row[i]
            if self._columns[name] is bool:
                val = bool(val)
            result[name] = val
        return result

    def _serialize_value(self, value: Any, column: str) -> Any:
        """Serialize a single value for a column."""
        if self._columns[column] is bool:
            return int(value)
        return value

    def _build_where(self, kwargs: dict[str, Any]) -> tuple[str, list[Any]]:
        """
        Parse Django-style lookup kwargs into a SQL WHERE clause and params.

        Returns (where_sql, params). where_sql is empty string if kwargs is
        empty, in which case callers should omit the WHERE clause entirely.

        Raises:
            KeyError: If a column name is unknown.
            ValueError: If a lookup suffix is unrecognised.
        """
        if not kwargs:
            return "", []

        conditions: list[str] = []
        params: list[Any] = []

        for key, value in kwargs.items():
            # Split "column__op" from the right so column names containing
            # underscores (e.g., "created_at") are handled correctly.
            if "__" in key:
                col, op_name = key.rsplit("__", 1)
                if op_name not in _LOOKUP_OPS:
                    raise ValueError(
                        f"Unknown lookup: {op_name!r}. "
                        f"Supported: {sorted(_LOOKUP_OPS.keys())}"
                    )
                sql_op = _LOOKUP_OPS[op_name]
            else:
                col = key
                sql_op = "="

            self._validate_column_name(col)
            conditions.append(f"{col} {sql_op} ?")
            params.append(self._serialize_value(value, col))

        return " AND ".join(conditions), params

    # -------------------------------------------------------------------------
    # Core CRUD
    # -------------------------------------------------------------------------

    @property
    def columns(self) -> dict[str, type]:
        """Return column definitions."""
        return dict(self._columns)

    def insert(self, record: dict[str, Any]) -> None:
        """Insert a single record. All columns must be present."""
        self._check_writable()
        self._validate_record(record)
        placeholders = ", ".join("?" for _ in self._col_names)
        col_sql = ", ".join(self._col_names)
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO {self._table_name} ({col_sql}) VALUES ({placeholders})",
                self._serialize(record),
            )
            conn.commit()

    def insert_many(self, records: list[dict[str, Any]]) -> None:
        """Insert multiple records in a single transaction."""
        self._check_writable()
        if not records:
            return
        for record in records:
            self._validate_record(record)
        placeholders = ", ".join("?" for _ in self._col_names)
        col_sql = ", ".join(self._col_names)
        with self._connect() as conn:
            conn.executemany(
                f"INSERT INTO {self._table_name} ({col_sql}) VALUES ({placeholders})",
                [self._serialize(r) for r in records],
            )
            conn.commit()

    def get(self, column: str, value: Any) -> list[dict[str, Any]]:
        """Get all records where column equals value."""
        self._validate_column_name(column)
        col_sql = ", ".join(self._col_names)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {col_sql} FROM {self._table_name} WHERE {column} = ?",
                (self._serialize_value(value, column),),
            ).fetchall()
            return [self._deserialize_row(row) for row in rows]

    def get_all(self) -> list[dict[str, Any]]:
        """Get all records."""
        col_sql = ", ".join(self._col_names)
        with self._connect() as conn:
            rows = conn.execute(f"SELECT {col_sql} FROM {self._table_name}").fetchall()
            return [self._deserialize_row(row) for row in rows]

    def find(self, **kwargs: Any) -> list[dict[str, Any]]:
        """
        Find records matching all given conditions (AND).

        Parses Django-style field lookups from keyword argument names:
          - No suffix -> equality:          name="Alice"
          - __gt      -> greater than:       score__gt=90.0
          - __lt      -> less than:          score__lt=90.0
          - __gte     -> greater or equal:   score__gte=80.0
          - __lte     -> less or equal:      score__lte=95.0
          - __ne      -> not equal:          active__ne=True

        With no arguments, returns all records (same as get_all()).

        Raises:
            KeyError: If a column name is unknown.
            ValueError: If a lookup suffix is unrecognised.
        """
        where_sql, params = self._build_where(kwargs)
        if not where_sql:
            return self.get_all()

        col_sql = ", ".join(self._col_names)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {col_sql} FROM {self._table_name} WHERE {where_sql}",
                params,
            ).fetchall()
            return [self._deserialize_row(row) for row in rows]

    def get_column(self, column: str, **kwargs: Any) -> list[Any]:
        """
        Get all values for a single column, with optional filtering.

        Accepts the same Django-style lookup kwargs as find().

        Examples:
            t.get_column("name")                   # all names
            t.get_column("name", active=True)       # names where active
            t.get_column("id", score__gt=90.0)      # ids with high score
        """
        self._validate_column_name(column)
        is_bool = self._columns[column] is bool
        where_sql, params = self._build_where(kwargs)
        query = f"SELECT {column} FROM {self._table_name}"
        if where_sql:
            query += f" WHERE {where_sql}"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            if is_bool:
                return [bool(row[0]) for row in rows]
            return [row[0] for row in rows]

    def update(
        self, where_column: str, where_value: Any, updates: dict[str, Any]
    ) -> int:
        """Update records where column equals value. Returns count updated."""
        self._check_writable()
        self._validate_column_name(where_column)
        for name, val in updates.items():
            self._validate_column_name(name)
            typ = self._columns[name]
            if not isinstance(val, typ):
                if typ is float and isinstance(val, int):
                    continue
                raise TypeError(
                    f"Column {name!r} expects {typ.__name__}, got {type(val).__name__}"
                )
        set_parts = [f"{name} = ?" for name in updates]
        set_sql = ", ".join(set_parts)
        params = [self._serialize_value(v, k) for k, v in updates.items()]
        params.append(self._serialize_value(where_value, where_column))
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE {self._table_name} SET {set_sql} WHERE {where_column} = ?",
                params,
            )
            conn.commit()
            return cursor.rowcount

    def delete(self, where_column: str, where_value: Any) -> int:
        """Delete records where column equals value. Returns count deleted."""
        self._check_writable()
        self._validate_column_name(where_column)
        with self._connect() as conn:
            cursor = conn.execute(
                f"DELETE FROM {self._table_name} WHERE {where_column} = ?",
                (self._serialize_value(where_value, where_column),),
            )
            conn.commit()
            return cursor.rowcount

    def clear(self) -> int:
        """Delete all records. Returns count deleted."""
        self._check_writable()
        with self._connect() as conn:
            cursor = conn.execute(f"DELETE FROM {self._table_name}")
            conn.commit()
            return cursor.rowcount

    # -------------------------------------------------------------------------
    # Iteration
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        """Return number of records."""
        with self._connect() as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM {self._table_name}").fetchone()
            return row[0]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all records."""
        return iter(self.get_all())

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------

    def to_csv(self, path: str | Path) -> None:
        """
        Export all records to a CSV file.

        Format: header row with column names, then data rows.
        """
        records = self.get_all()
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._col_names)
            writer.writeheader()
            writer.writerows(records)

    def to_json(self, path: str | Path, indent: int | None = 2) -> None:
        """
        Export all records to a JSON file.

        Format: [{"col1": val1, ...}, ...]
        """
        records = self.get_all()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=indent, ensure_ascii=False)

    # -------------------------------------------------------------------------
    # Import
    # -------------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        db_path: str | Path,
        csv_path: str | Path,
        columns: dict[str, type],
        **kwargs: Any,
    ) -> SQLiteTableStore:
        """
        Create a SQLiteTableStore from a CSV file.

        Expects a header row matching column names.

        Args:
            db_path: Path to create the database at.
            csv_path: Path to CSV file.
            columns: Column definitions.
            **kwargs: Additional arguments (table_name, primary_key).
        """
        store = cls(db_path, columns, **kwargs)
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = []
            for row in reader:
                record: dict[str, Any] = {}
                for name, typ in columns.items():
                    raw = row[name]
                    if typ is bool:
                        record[name] = raw.lower() in ("true", "1")
                    elif typ is int:
                        record[name] = int(raw)
                    elif typ is float:
                        record[name] = float(raw)
                    else:
                        record[name] = raw
                records.append(record)
            if records:
                store.insert_many(records)
        return store

    @classmethod
    def from_json(
        cls,
        db_path: str | Path,
        json_path: str | Path,
        columns: dict[str, type],
        **kwargs: Any,
    ) -> SQLiteTableStore:
        """
        Create a SQLiteTableStore from a JSON file.

        Expects format: [{"col1": val1, ...}, ...]

        Args:
            db_path: Path to create the database at.
            json_path: Path to JSON file.
            columns: Column definitions.
            **kwargs: Additional arguments (table_name, primary_key).
        """
        store = cls(db_path, columns, **kwargs)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data:
            store.insert_many(data)
        return store

    # -------------------------------------------------------------------------
    # Context manager & close
    # -------------------------------------------------------------------------

    def close(self) -> None:
        """
        Explicitly close the store and checkpoint the WAL.

        This is optional - connections are closed after each operation.
        However, calling close() forces a WAL checkpoint which can be
        useful before copying the database file.

        In read-only mode, this is a no-op.
        """
        if self.readonly:
            return
        with self._connect() as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    # -------------------------------------------------------------------------
    # Repr
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        mode = "readonly" if self.readonly else "rw"
        return (
            f"SQLiteTableStore({self.path!r}, "
            f"table={self._table_name!r}, {mode}, len={len(self)})"
        )
