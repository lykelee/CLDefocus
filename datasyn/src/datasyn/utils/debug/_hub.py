from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any


class _Box:
    """Attribute-style access to hub storage."""

    def __init__(self, store: dict[str, Any]) -> None:
        object.__setattr__(self, "_store", store)

    def __getattr__(self, name: str) -> Any:
        store = object.__getattribute__(self, "_store")
        if name not in store:
            raise AttributeError(f"Hub has no key {name!r}")
        return store[name]

    def __setattr__(self, name: str, value: Any) -> None:
        object.__getattribute__(self, "_store")[name] = value

    def __delattr__(self, name: str) -> None:
        store = object.__getattribute__(self, "_store")
        if name not in store:
            raise AttributeError(f"Hub has no key {name!r}")
        del store[name]

    def __dir__(self) -> list[str]:
        store = object.__getattribute__(self, "_store")
        return list(store.keys())

    def __repr__(self) -> str:
        store = object.__getattribute__(self, "_store")
        if not store:
            return "<Box (empty)>"
        items = "  ".join(f"{k}={_truncr(v)}" for k, v in store.items())
        return f"<Box  {items}>"


def _truncr(obj: Any, max_len: int = 50) -> str:
    r = repr(obj)
    return r if len(r) <= max_len else r[:max_len] + "..."


def _safe_deepcopy(obj: Any) -> Any:
    try:
        return copy.deepcopy(obj)
    except Exception:
        return obj


def _safe_eq(a: Any, b: Any) -> bool:
    """Return True if a and b are equal, handling types where == returns non-bool."""
    try:
        result = a == b
        # numpy/torch arrays return array-like results
        if hasattr(result, "__bool__"):
            return bool(result)
        if hasattr(result, "all"):
            return bool(result.all())
        return bool(result)
    except Exception:
        return False


class Hub:
    """Global state teleportation — share variables across stack frames."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._snapshots: dict[str, Any] = {}
        self.box = _Box(self._store)

    def put(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def dump_to_disk(self, path: str | Path = "hub_dump.pkl") -> None:
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump(self._store, f)
        print(f"[hub] dumped {len(self._store)} keys -> {path}")

    def load_from_disk(self, path: str | Path = "hub_dump.pkl") -> None:
        path = Path(path)
        with open(path, "rb") as f:
            loaded: dict[str, Any] = pickle.load(f)
        self._store.update(loaded)
        print(f"[hub] loaded {len(loaded)} keys <- {path}")

    def diff(self, key: str) -> None:
        """Show how a stored value changed since the last diff() call."""
        if key not in self._store:
            print(f"[hub.diff] key {key!r} not found")
            return
        current = self._store[key]
        if key not in self._snapshots:
            self._snapshots[key] = _safe_deepcopy(current)
            print(f"[hub.diff] {key!r}: snapshot taken (first call)")
            return
        prev = self._snapshots[key]
        if _safe_eq(prev, current):
            print(f"[hub.diff] {key!r}: no change")
        else:
            print(f"[hub.diff] {key!r} changed:")
            print(f"  before: {prev!r}")
            print(f"  after : {current!r}")
        self._snapshots[key] = _safe_deepcopy(current)

    def show(self) -> None:
        """Pretty-print all stored key-value pairs."""
        if not self._store:
            print("[hub] (empty)")
            return
        width = max(len(k) for k in self._store)
        print(f"[hub] {len(self._store)} item(s):")
        for k, v in self._store.items():
            print(f"  {k:<{width}} : {_truncr(v, max_len=120)}")

    def keys(self) -> list[str]:
        return list(self._store.keys())

    def clear(self) -> None:
        self._store.clear()
        self._snapshots.clear()

    def __repr__(self) -> str:
        if not self._store:
            return "<Hub (empty)>"
        items = "  ".join(f"{k}={_truncr(v)}" for k, v in self._store.items())
        return f"<Hub  {items}>"
