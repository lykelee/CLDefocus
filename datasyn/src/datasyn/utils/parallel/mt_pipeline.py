from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

T = TypeVar("T")
U = TypeVar("U")

_SENTINEL = object()
Item = Union[object, Tuple[int, Any]]  # either _SENTINEL or (seq, payload)


@dataclass(frozen=True)
class Stage:
    fn: Callable[[Any], Any]
    workers: int = 1
    name: str = ""


class ThreadPipeline:
    """
    A simple N-stage pipeline:
      inputs -> stage1 -> stage2 -> ... -> stageN -> outputs

    - Each stage has `workers` threads.
    - Queues between stages can be bounded (backpressure).
    - Carries (seq, payload) through the pipeline so output can be re-ordered if desired.
    - One-in / one-out per stage function.

    For advanced funtionalities, consider using the following (**not verified yet**):
    - **Pypeln**: explicitly models a multi-stage pipeline; each stage can use **threads, processes, or asyncio tasks**, with queues between stages.
    - **Streamz**: streaming pipeline framework (more “stream processing”; supports backpressure, branching, etc.).
    - **Dask**: great when you can express the work as a task graph / delayed calls (less “queue pipeline”, more “scheduler”).
    - **Ray**: for scaling beyond one machine / larger distributed workloads (also has “streaming execution” in Ray Data contexts).
    """

    def __init__(
        self,
        stages: List[Stage],
        *,
        queue_maxsize: int = 0,  # 0 => unbounded
        preserve_order: bool = False,
    ) -> None:
        if not stages:
            raise ValueError("stages must be non-empty")
        if any(s.workers <= 0 for s in stages):
            raise ValueError("stage workers must be >= 1")

        self._stages = stages
        self._queue_maxsize = queue_maxsize
        self._preserve_order = preserve_order

        self._stop_event = threading.Event()
        self._err_lock = threading.Lock()
        self._first_error: Optional[BaseException] = None

    def map(self, inputs: Iterable[T]) -> Iterator[Any]:
        qs: List[queue.Queue[Item]] = [
            queue.Queue(maxsize=self._queue_maxsize)
            for _ in range(len(self._stages) + 1)
        ]
        threads: List[threading.Thread] = []

        # Feeder: enumerate inputs and put (seq, x) into q0, then one sentinel.
        def feeder() -> None:
            try:
                for seq, x in enumerate(inputs):
                    if self._stop_event.is_set():
                        break
                    qs[0].put((seq, x))
            except BaseException as e:
                self._set_error(e)
            finally:
                # Start shutdown chain.
                qs[0].put(_SENTINEL)

        threads.append(
            threading.Thread(target=feeder, name="pipeline:feeder", daemon=True)
        )

        # Stage workers
        for stage_idx, stage in enumerate(self._stages):
            in_q = qs[stage_idx]
            out_q = qs[stage_idx + 1]

            stopped = 0
            stopped_lock = threading.Lock()

            def make_worker(
                si: int, st: Stage, inq: queue.Queue[Item], outq: queue.Queue[Item]
            ) -> Callable[[], None]:
                def worker() -> None:
                    nonlocal stopped
                    while True:
                        item = inq.get()
                        if item is _SENTINEL:
                            # Let all workers see the sentinel; only the last forwards it downstream.
                            with stopped_lock:
                                stopped += 1
                                is_last = stopped == st.workers
                            if not is_last:
                                inq.put(_SENTINEL)  # re-insert for remaining workers
                            else:
                                outq.put(_SENTINEL)  # forward once
                            return

                        seq, payload = item  # type: ignore[misc]
                        if self._stop_event.is_set():
                            continue
                        try:
                            out = st.fn(payload)
                        except BaseException as e:
                            self._set_error(e)
                            # Kick off shutdown; best-effort (don’t block forever).
                            try:
                                inq.put_nowait(_SENTINEL)
                            except queue.Full:
                                pass
                            try:
                                outq.put_nowait(_SENTINEL)
                            except queue.Full:
                                pass
                            return
                        outq.put((seq, out))

                return worker

            for w in range(stage.workers):
                name = (
                    f"pipeline:stage{stage_idx}:{stage.name or stage.fn.__name__}:{w}"
                )
                threads.append(
                    threading.Thread(
                        target=make_worker(stage_idx, stage, in_q, out_q),
                        name=name,
                        daemon=True,
                    )
                )

        # Start all threads
        for t in threads:
            t.start()

        # Consumer generator
        def output_iter() -> Iterator[Any]:
            try:
                if not self._preserve_order:
                    while True:
                        item = qs[-1].get()
                        if item is _SENTINEL:
                            break
                        seq, out = item  # type: ignore[misc]
                        yield out
                else:
                    next_seq = 0
                    buf: Dict[int, Any] = {}
                    while True:
                        item = qs[-1].get()
                        if item is _SENTINEL:
                            break
                        seq, out = item  # type: ignore[misc]
                        buf[seq] = out
                        while next_seq in buf:
                            yield buf.pop(next_seq)
                            next_seq += 1
            finally:
                # Ensure threads exit; surface error if any.
                for t in threads:
                    t.join()
                if self._first_error is not None:
                    raise self._first_error

        return output_iter()

    def _set_error(self, e: BaseException) -> None:
        with self._err_lock:
            if self._first_error is None:
                self._first_error = e
                self._stop_event.set()
