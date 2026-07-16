"""Small spawn-based subprocess helpers.

This module intentionally imports no GPU/JAX-related packages.  It is useful
when code should run outside the parent process only to isolate imports or
environment variables, not to distribute work across GPUs.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

TRet = TypeVar("TRet")


def _subprocess_entry(
    queue,
    func: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    env: Mapping[str, str | None],
) -> None:
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    try:
        queue.put(("ok", func(*args, **dict(kwargs))))
    except BaseException as exc:  # noqa: BLE001 - must cross process boundary.
        queue.put(("err", repr(exc), traceback.format_exc()))


def run_single_subprocess(
    func: Callable[..., TRet],
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    env: Mapping[str, str | None] | None = None,
) -> TRet:
    """Run ``func`` once in a spawned subprocess and return its result.

    Args:
        func: Picklable callable to run in the child process.
        args: Positional arguments for ``func``.
        kwargs: Keyword arguments for ``func``.
        env: Environment overrides applied only in the child.  A value of
            ``None`` deletes that environment variable in the child.

    Raises:
        RuntimeError: If the child raises or exits without returning a result.
    """
    kwargs = kwargs or {}
    env = env or {}

    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_subprocess_entry,
        args=(queue, func, tuple(args), dict(kwargs), dict(env)),
    )
    proc.start()
    proc.join()

    if queue.empty():
        raise RuntimeError(
            f"Subprocess exited without returning a result; exitcode={proc.exitcode}"
        )

    message = queue.get()
    if message[0] == "ok":
        return message[1]

    _, exc_repr, tb = message
    raise RuntimeError(f"Subprocess raised {exc_repr}\n{tb}")

