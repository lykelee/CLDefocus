"""
WARNING: In this module, modules related to GPUs (e.g., PyTorch) must not globally imported!
"""

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Concatenate,
    Iterator,
    Optional,
    ParamSpec,
    Sequence,
    TypeVar,
)

from datasyn.utils.parallel.easy_tqdm import easy_tqdm
from datasyn.utils.parallel.gpu_pool import Context, GPUProcessPool, Output

TData = TypeVar("TData")
TIndex = TypeVar("TIndex")
TRet = TypeVar("TRet")
P = ParamSpec("P")


@dataclass
class MPPerGPUConfig:
    worker_func: Callable[Concatenate[Context[TData], TIndex, P], TRet]
    indices: Sequence[TIndex]
    worker_args: Sequence = ()
    fn_init_proc: Optional[Callable[[int], Any]] = None
    init_args: Sequence = ()
    force_mp: bool = (
        False  # If True, use MP even if only one GPU, might be used for debugging
    )
    show_progress: bool = True
    """Show an inner easy_tqdm bar over task completion.

    Set to False when the caller renders its own progress (e.g., a
    ``PipelineProgressMonitor``) — two bars at the same terminal position
    will overwrite each other and flicker.
    """


def get_visible_gpu_ids():
    """
    Get visible CUDA device ids.
    It's actually numbers set in `CUDA_VISIBLE_DEVICES`.
    """
    import os

    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is None or env == "":  # Fallback to "0..N-1"
        import jax

        n = len(jax.devices("gpu"))
        return list(range(n))
    else:
        return [int(x) for x in env.split(",")]


def _temp_err_handling(out: Output[TIndex, TRet]):
    from termcolor import colored

    if out.err is not None:
        print(colored(70 * "=", "light_red"))
        print(colored("Exception in a subprocess!", "light_red"))
        print(
            colored("Task index: ", "light_yellow")
            + colored(f"{out.idx}", "light_green")
        )
        print(
            colored("Device id: ", "light_yellow")
            + colored(f"{out.device}", "light_green")
        )
        print(colored(f"Exception: {repr(out.err.exc)}", "light_red"))
        print(colored(f"Trackback:\n{out.err.tb}", "light_red"))
        print(colored(70 * "=", "light_red"))

        raise out.err.exc


def run_mp_per_gpu_iter(cfg: MPPerGPUConfig) -> Iterator[TRet]:
    """Like :func:`run_mp_per_gpu`, but yields results as workers complete.

    Useful for callers that want to persist per-task progress incrementally
    (e.g., marking pipeline items done in SQLite) or render their own
    progress bar without waiting for the whole batch.

    Yields each worker's return value in completion order (not input order).
    """
    gpu_ids = get_visible_gpu_ids()
    n_gpus = len(gpu_ids)
    assert n_gpus >= 1

    indices = cfg.indices
    n_tasks = len(indices)

    if n_gpus == 1 and not cfg.force_mp:
        device = gpu_ids[0]

        ctx = Context(device=device, id=0, data=None, is_pooled=False)

        if cfg.fn_init_proc is None:
            data = None
        else:
            data = cfg.fn_init_proc(ctx, *cfg.init_args)

        ctx = Context(device=device, id=0, data=data, is_pooled=False)

        desc = f"No MP (n_gpus={n_gpus})"
        loop: Any = indices
        if cfg.show_progress:
            loop = easy_tqdm(loop, desc=desc, total=n_tasks)
        for idx in loop:
            ret = cfg.worker_func(ctx, idx, *cfg.worker_args)
            yield ret

    else:
        with GPUProcessPool(
            devices=gpu_ids,
            make_data=cfg.fn_init_proc,
            init_args=cfg.init_args,
        ) as pool:
            loop = pool.imap_unordered(cfg.worker_func, indices, cfg.worker_args)

            desc = f"MP per GPUs (n_gpus={n_gpus})"
            if cfg.show_progress:
                loop = easy_tqdm(loop, desc=desc, total=n_tasks)

            for out in loop:
                _temp_err_handling(out)
                if out.ret is not None:
                    yield out.ret


def run_mp_per_gpu(cfg: MPPerGPUConfig) -> list[TRet]:
    """
    If only one GPU is available, this doesn't use multiprocessing and just call the worker function.

    Collects all results into a list.  Use :func:`run_mp_per_gpu_iter` if
    you want streaming (per-task) results.
    """
    # TODO: Return value might be important in the future!
    return list(run_mp_per_gpu_iter(cfg))
