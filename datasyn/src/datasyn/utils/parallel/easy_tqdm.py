import inspect
import threading
from typing import Iterable, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")

# thread‐local storage so that different threads have independent nesting levels
_local = threading.local()


def try_find_length(iterable: Iterable):
    # TODO: Currently, this is meaningless!
    if hasattr(iterable, "__len__"):
        return len(iterable)

    return None  # Failed


def make_caller_desc(depth: int = 1):
    frame = inspect.currentframe()

    caller_frame = frame.f_back  # Caller of this function
    for _ in range(depth):
        caller_frame = caller_frame.f_back

    info = inspect.getframeinfo(caller_frame)
    desc = f"{info.filename}:{info.lineno}"

    return desc


def easy_tqdm(
    iterable: Iterable[T] = None,
    disable: bool = False,
    find_length: bool = True,
    **kwargs,
):
    """
    Like tqdm, but auto-detects nesting level for position/leave.

    TODO: Consider thread-safety!
    """
    # TODO: This is quick-and-dirty for disabled tqdm in multithreading!
    if disable:
        return tqdm(iterable, disable=True)

    # grab current nesting depth (0 = root)
    level = getattr(_local, "level", 0)

    # set position = nesting level
    kwargs.setdefault("position", level)
    # leave only if root bar
    kwargs.setdefault("leave", level == 0)

    # Set default description
    desc_caller = make_caller_desc()
    kwargs.setdefault("desc", desc_caller)

    # bump nesting for any inner calls
    _local.level = level + 1

    if find_length:
        total = try_find_length(iterable)
        kwargs.setdefault("total", total)

    bar = tqdm(iterable, **kwargs)

    # wrap its close() so we can decrement our counter exactly once
    orig_close = bar.close
    closed = False

    def close_once():
        nonlocal closed
        if not closed:
            closed = True
            orig_close()
            # back out nesting bump
            _local.level -= 1

    bar.close = close_once
    return bar
