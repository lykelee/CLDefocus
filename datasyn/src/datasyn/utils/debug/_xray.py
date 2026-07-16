from __future__ import annotations

import inspect
from typing import Any


def see(obj: Any, private: bool = False) -> dict[str, Any]:
    """
    Dump all non-callable attributes of obj.

    Args:
        obj:     The object to inspect.
        private: If True, include single-underscore attributes.
                 Dunder attributes are always excluded.
    """
    result: dict[str, Any] = {}
    for name in dir(obj):
        if name.startswith("__"):
            continue
        if not private and name.startswith("_"):
            continue
        try:
            val = getattr(obj, name)
        except Exception as e:
            result[name] = f"<error: {e}>"
            continue
        if inspect.isroutine(val):
            continue
        result[name] = val

    lines = [f"<{type(obj).__qualname__}>"]
    for k, v in result.items():
        r = repr(v)
        if len(r) > 120:
            r = r[:120] + "..."
        lines.append(f"  {k}: {r}")
    print("\n".join(lines))
    return result


def structure(obj: Any, depth: int = 3) -> str:
    """
    Print the shape/structure of data without dumping its full contents.

    Args:
        obj:   The object to inspect.
        depth: How many nesting levels to expand (default 3).

    Returns:
        The structure string (also printed to stdout).
    """
    result = _structure(obj, depth, 0)
    print(result)
    return result


def _structure(obj: Any, depth: int, current: int) -> str:
    if current >= depth:
        return f"{type(obj).__name__}(...)"

    if isinstance(obj, dict):
        if not obj:
            return "dict{}"
        k, v = next(iter(obj.items()))
        return (
            f"dict[{type(k).__name__} -> {_structure(v, depth, current + 1)}]"
            f" len={len(obj)}"
        )

    if isinstance(obj, list):
        if not obj:
            return "list[]"
        return f"list[{_structure(obj[0], depth, current + 1)}] len={len(obj)}"

    if isinstance(obj, tuple):
        if not obj:
            return "tuple()"
        parts = [_structure(x, depth, current + 1) for x in obj[:4]]
        if len(obj) > 4:
            parts.append("...")
        return f"tuple({', '.join(parts)}) len={len(obj)}"

    if isinstance(obj, (set, frozenset)):
        t = type(obj).__name__
        if not obj:
            return f"{t}{{}}"
        sample = next(iter(obj))
        return f"{t}[{type(sample).__name__}] len={len(obj)}"

    # numpy / torch tensors and anything with .shape + .dtype
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        try:
            shape = list(obj.shape)
            dtype = str(obj.dtype)
            return f"{type(obj).__name__} shape={shape} dtype={dtype}"
        except Exception:
            pass

    if isinstance(obj, str):
        preview = repr(obj[:50] + "...") if len(obj) > 50 else repr(obj)
        return f"str(len={len(obj)}, {preview})"

    if isinstance(obj, bytes):
        return f"bytes(len={len(obj)})"

    r = repr(obj)
    if len(r) > 80:
        r = r[:80] + "..."
    return r
