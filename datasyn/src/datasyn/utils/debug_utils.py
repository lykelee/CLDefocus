"""
A quick and dirty way to store global values across frames during debugging.
"""

import datetime
import inspect
from typing import Any, Dict, Optional

_store: Dict[Any, Dict[Any, Any]] = {}


def put(key: Any, value: Any) -> Any:
    """
    Store `value` under `key`, along with timestamp & calling-frame info.
    Returns the value for easy in-line usage.
    """
    # Grab the caller’s frame
    caller = inspect.currentframe().f_back
    info = {
        "timestamp": datetime.datetime.now(),
        "filename": caller.f_code.co_filename,
        "lineno": caller.f_lineno,
        "func": caller.f_code.co_name,
    }
    _store[key] = {"value": value, "info": info}
    return value


def get(key: Any, default: Optional[Any] = None) -> Any:
    """
    Retrieve the value stored under `key`, or `default` if not present.
    """
    entry = _store.get(key)
    return entry["value"] if entry else default


def ref(key: Any, value: Optional[Any] = None) -> Any:
    """
    Put and get with one function!
    For simple usages, we can do everything only with this function.
    """
    if value is None:
        return get(key)
    else:
        return put(key, value), info(key)


def info(key: Any) -> Optional[Dict[Any, Any]]:
    """
    Retrieve the metadata (timestamp, file, line, func) for `key`.
    """
    entry = _store.get(key)
    return entry["info"] if entry else None


def clear(key: Optional[Any] = None) -> None:
    """
    Remove one key (if given) or clear the entire store.
    """
    if key is None:
        _store.clear()
    else:
        _store.pop(key, None)


def keys() -> list:
    """List all stored keys."""
    return list(_store.keys())


_counter = 0


def counter() -> int:
    global _counter
    v = _counter
    _counter += 1
    return v
