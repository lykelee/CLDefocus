"""
WARNING: In this module, modules related to GPUs (e.g., PyTorch) must not globally imported!

TODO: Reorganize this tip later!
[Tip]
Using GPU pool requires a lot of care.
You should ensure that no modules that use CUDA (e.g., JAX) are imported at top-level of the following modules:
- The main module that runs a pool.
- The worker module.
- All modules that define types of the arguments passed to workers.
  For example, if you define a class in a module and sends its instances to workers, that module belongs to this category.
If one of them is not satisfied, CUDA might be initialized before the pool distributes GPUs to workers.
"""

import multiprocessing as mp
import os
import traceback
from dataclasses import dataclass
from functools import partial
from typing import (
    Callable,
    Concatenate,
    Generic,
    Iterable,
    Optional,
    ParamSpec,
    Sequence,
    TypeVar,
)

TData = TypeVar("TData")
TIndex = TypeVar("TIndex")
TRet = TypeVar("TRet")
P = ParamSpec("P")


@dataclass(frozen=True)
class Context(Generic[TData]):
    device: int  # Device ID
    id: int  # ID of the subprocess starting from 0
    data: TData
    is_pooled: bool  # False if it's actually not in a pooled process


_CTX: Context = None  # Per-process context (data, device_id, etc.)


_INIT_LOCK_TIMEOUT = 600  # seconds; long enough for any realistic make_data


def _worker_init(
    counter, lock, init_lock, devices, make_data, init_args: Sequence, setup_gpu
):
    """
    Initializer run once per worker process.

    devices: sequence of device ids (e.g., [0, 1, 2])
    make_data: callable(device_id) -> data handle
    setup_gpu: callable(device_id) -> None (optional extra setup)
    """
    global _CTX

    # Each worker grabs a unique index into `devices`
    with lock:
        idx = counter.value
        counter.value += 1

    device = devices[idx]

    # Set env before importing / using CUDA frameworks
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    # Configure logging for this subprocess.  Spawned processes inherit no
    # handlers, so basicConfig(force=True) installs a stderr StreamHandler.
    # Every message is tagged with the GPU id and pid for easy filtering.
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [GPU {device} pid {os.getpid()}] %(name)s: %(message)s",
        force=True,
    )
    _log = logging.getLogger("gpu_pool")
    _log.setLevel(logging.WARNING)
    _log.info(f"Worker {idx} started (CUDA_VISIBLE_DEVICES={device})")

    if setup_gpu is not None:
        setup_gpu(device)

    ctx = Context(device=device, id=idx, data=None, is_pooled=True)

    # Serialise CUDA context initialisation across workers.
    # All workers are spawned simultaneously by mp.Pool, so without this lock
    # they would all call make_data (which triggers JAX/CUDA context creation
    # and memory pre-allocation) concurrently.  The CUDA driver is not stable
    # under concurrent multi-process context creation and produces
    # non-deterministic errors that disappear on retry.
    # Holding init_lock for the duration of make_data guarantees that each
    # worker fully completes its CUDA initialisation before the next one begins.
    #
    # We use acquire(timeout=...) rather than `with init_lock` so that a hard
    # crash in another worker (which would leave the lock permanently held)
    # turns into a loud error instead of a silent infinite hang.
    _log.info(f"Worker {idx}: waiting for init_lock")
    acquired = init_lock.acquire(timeout=_INIT_LOCK_TIMEOUT)
    if not acquired:
        raise RuntimeError(
            f"Worker {idx} (GPU {device}) timed out after {_INIT_LOCK_TIMEOUT}s "
            "waiting for init_lock.  Another worker likely crashed while holding it."
        )
    _log.info(f"Worker {idx}: acquired init_lock, running make_data")
    try:
        data = make_data(ctx, *init_args) if make_data is not None else None
    finally:
        init_lock.release()
        _log.info(f"Worker {idx}: released init_lock.")

    _CTX = Context(device=device, id=idx, data=data, is_pooled=True)
    _log.info(f"Worker {idx}: init complete.")

    # TODO: This is quick-and-dirty! Revise it later!
    # Ensure data gets closed on process exit
    import atexit

    def _cleanup():
        close = getattr(_CTX.data, "close", None)
        if callable(close):
            close()

    atexit.register(_cleanup)


@dataclass(frozen=True)
class Err:
    exc: Exception
    tb: str


@dataclass(frozen=True)
class Output(Generic[TIndex, TRet]):
    device: int
    subproc_id: int
    idx: TIndex
    ret: Optional[TRet]
    err: Optional[Err]


def _entry(
    func: Callable[Concatenate[Context[TData], TIndex, P], TRet],
    args: Sequence,
    idx: TIndex,
) -> Output[TIndex, TRet]:
    """
    Real worker entrypoint used by Pool.
    job = (user_func, args, kwargs)

    This catches exceptions raised in subprocesses and returns it as the output.
    Thus, this function is exception-safe.
    """
    err = None

    try:
        ret = func(_CTX, idx, *args)

    except Exception as e:
        ret = None
        tb = traceback.format_exc()
        err = Err(exc=e, tb=tb)

    return Output(device=_CTX.device, subproc_id=_CTX.id, idx=idx, ret=ret, err=err)


def entry_fake(
    func: Callable[Concatenate[Context[TData], TIndex, P], TRet],
    args: Sequence,
    idx: TIndex,
    ctx: Context[TData],
):
    """
    This is used to mimic the entry function with no pooling.

    TODO: This looks dirty! Make a better idea if possible.
    """
    global _CTX
    _CTX = ctx
    return _entry(func, args, idx)


class GPUProcessPool:
    def __init__(self, devices, make_data, init_args: Sequence, setup_gpu=None):
        """
        devices: list of CUDA device indices, e.g., [0, 1, 3]
        make_data: callable(device_id) -> data handle for this process
        setup_gpu: optional callable(device_id) to do GPU-only init
                   (e.g., torch.cuda.set_device)
        """
        ctx = mp.get_context("spawn")
        counter = ctx.Value("i", 0)
        lock = ctx.Lock()
        init_lock = ctx.Lock()

        self._pool = ctx.Pool(
            processes=len(devices),
            initializer=_worker_init,
            initargs=(
                counter,
                lock,
                init_lock,
                devices,
                make_data,
                init_args,
                setup_gpu,
            ),
        )

    def imap_unordered(self, func, iterable: Iterable, args: Sequence, chunksize=1):
        return self._pool.imap_unordered(
            partial(_entry, func, args), iterable, chunksize
        )

    def close(self):
        self._pool.close()

    def join(self):
        self._pool.join()

    def terminate(self):
        self._pool.terminate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        self.join()
