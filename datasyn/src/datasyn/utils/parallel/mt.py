"""
Multi-threading

TODO: The below is important note, but temporarily written here.
      We need to move this to good place!
- File-based (e.g., images, numpy arrays): Suitable with multiple readers/writers
- Many structured data (e.g., HDF5, LMDB, Zarr): Good for multiple readers, good for one writer
"""

import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Optional, TypeVar

TIndex = TypeVar("TIndex")
TOutput = TypeVar("TOutput")
TSetup = TypeVar("TSetup")


def _safe_execute(fun: Callable[[], TOutput]):
    try:
        out = fun()

    except BaseException as e:
        out = None
        tb = traceback.format_exc()
        print(tb)

    return out


def run_thread_pools(
    fun: Callable[..., TOutput],
    indices: Iterable[TIndex],
    parallel: int = 0,
    fn_setup: Optional[Callable[[], TSetup]] = None,
    tqdm_kwargs: dict = {},
):
    from datasyn.utils.parallel.easy_tqdm import easy_tqdm, try_find_length

    use_setup = fn_setup is not None

    outs = list[TOutput | None]()

    if parallel == 0:  # No parallelization
        setup = fn_setup() if use_setup else None
        for idx in easy_tqdm(indices, **tqdm_kwargs):
            kwargs = dict()
            if use_setup:
                kwargs["setup"] = setup

            out = fun(idx, **kwargs)

            outs.append(out)

    elif parallel > 0:
        thread_ctx = threading.local()

        if "total" not in tqdm_kwargs:
            tqdm_kwargs["total"] = try_find_length(indices)

        with easy_tqdm(indices, **tqdm_kwargs) as pbar:

            def fun_wrapper(idx: TIndex):
                kwargs = dict()

                if use_setup:
                    if not hasattr(thread_ctx, "setup"):
                        thread_ctx.setup = fn_setup()
                    kwargs["setup"] = thread_ctx.setup

                out = fun(idx, **kwargs)

                pbar.update(1)  # tqdm is thread-safe.

                return out

            with ThreadPoolExecutor(max_workers=parallel) as ex:
                futures = {ex.submit(fun_wrapper, idx): idx for idx in indices}

                for f in as_completed(futures):
                    idx = futures[f]
                    out = _safe_execute(lambda: f.result())
                    outs.append(out)

    else:
        raise ValueError()

    return outs
