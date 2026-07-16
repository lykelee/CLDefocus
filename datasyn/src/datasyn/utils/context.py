from typing import TypeVar

T = TypeVar("T")


def maybe_ctx(ctx: T, enabled: bool):
    from contextlib import nullcontext

    return ctx if enabled else nullcontext()


def ctx_or_none(ctx: T):
    """If the given is None, returns nullcontext."""
    from contextlib import nullcontext

    return ctx if ctx is not None else nullcontext()

