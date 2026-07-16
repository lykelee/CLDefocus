"""
Lightweight per-process stage profiler (temporary; env-gated).

GPU stages (s5_psf_gen/s6_blur) run their per-item work inside short-lived per-GPU worker
subprocesses, so there is no single place to aggregate timings. Each worker
process instead owns one ``StageProf`` instance that accumulates wall time per
named phase and prints a running summary every ``every`` items. Output is
tagged with the GPU id + pid so concurrent workers can be told apart.

Pure-Python (no JAX/CUDA imports) so it is safe to import at module top level
in stages that must defer JAX. For JAX-async phases, call a device barrier
(e.g., ``jx.block_until_ready_all()``) inside the ``t(...)`` block so the
measured time reflects real compute, not just dispatch.

Enable with the stage's env flag, e.g., ``SYN_S5PROF=1`` / ``SYN_S6PROF=1``.
"""

from __future__ import annotations

import contextlib
import os
import time


class StageProf:
    def __init__(self, tag: str, env: str, every: int = 20) -> None:
        self.tag = tag
        self.enabled = False
        self.every = max(1, every)
        self.device: object = "?"
        self._acc: dict[str, float] = {}
        self._n = 0

    def add(self, key: str, dt: float) -> None:
        if self.enabled:
            self._acc[key] = self._acc.get(key, 0.0) + dt

    @contextlib.contextmanager
    def t(self, key: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(key, time.perf_counter() - t0)

    def item_done(self) -> None:
        """Mark one item processed; print a summary every ``every`` items."""
        if not self.enabled:
            return
        self._n += 1
        if self._n % self.every != 0:
            return
        parts = " | ".join(f"{k}={v:.2f}s" for k, v in sorted(self._acc.items()))
        tot = sum(self._acc.values())
        print(
            f"[{self.tag}] dev{self.device} pid{os.getpid()} n={self._n} "
            f"{parts} | sum~{tot:.1f}s",
            flush=True,
        )
