"""StageNode wrappers that run JAX-touching stages in child processes."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasyn.pipeline import ItemProgress, StageNode
from datasyn.utils.parallel import run_single_subprocess


def cpu_only_jax_env() -> dict[str, str]:
    """Environment overrides for child processes that must not touch GPUs."""
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORM_NAME": "cpu",
        "JAX_PLATFORMS": "cpu",
        # "JAX_SKIP_CUDA_CONSTRAINTS_CHECK": "1",
        # "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }


@dataclass(frozen=True)
class _NodeProxySpec:
    name: str
    store_dir: str
    deps: tuple["_NodeProxySpec", ...] = ()
    root_total: int | None = None
    selected_ids_store_relpath: str | None = None
    selection_complete_relpath: str | None = None


class _NodeStatusProxy:
    """Small dependency proxy with enough StageNode behavior for child runs."""

    def __init__(self, spec: _NodeProxySpec) -> None:
        self.name = spec.name
        self.store_dir = Path(spec.store_dir)
        self.deps = [_NodeStatusProxy(dep) for dep in spec.deps]
        self.progress = ItemProgress(self.store_dir / "_progress")
        self._root_total = spec.root_total
        self._selected_ids_store_path = (
            self.store_dir / spec.selected_ids_store_relpath
            if spec.selected_ids_store_relpath is not None
            else None
        )
        self._selection_complete_path = (
            self.store_dir / spec.selection_complete_relpath
            if spec.selection_complete_relpath is not None
            else None
        )

    def _selected_done_items(self) -> frozenset[str]:
        if (
            self._selected_ids_store_path is None
            or self._selection_complete_path is None
            or not self._selection_complete_path.exists()
        ):
            return frozenset()

        from datasyn.synthesis.common import IOMode
        from datasyn.synthesis.stores.patch_selection_store import PatchSelectionStore

        store = PatchSelectionStore(self._selected_ids_store_path, mode=IOMode.READ)
        return frozenset(store.list_names())

    def ready_items(self) -> list[str]:
        already_handled = self.progress.processable()

        if self._selected_ids_store_path is not None:
            if self.deps and not all(dep.is_complete() for dep in self.deps):
                return []
            return sorted(self._selected_done_items() - already_handled)

        if not self.deps:
            if self._root_total is None:
                return []
            return [] if self.is_complete() else ["__root_incomplete__"]

        dep_done_sets = [dep.progress.done_items() for dep in self.deps]
        upstream_done = dep_done_sets[0].intersection(*dep_done_sets[1:])
        return sorted(upstream_done - already_handled)

    def is_complete(self) -> bool:
        if self._selected_ids_store_path is not None:
            if self.deps and not all(dep.is_complete() for dep in self.deps):
                return False
            if (
                self._selection_complete_path is None
                or not self._selection_complete_path.exists()
            ):
                return False
            return not self.ready_items()

        if self._root_total is not None:
            return len(self.progress.done_items()) >= self._root_total
        if not self.deps:
            return len(self.ready_items()) == 0
        return all(dep.is_complete() for dep in self.deps) and not self.ready_items()

    def progress_total(self) -> int | None:
        if self._selected_ids_store_path is not None:
            if (
                self._selection_complete_path is None
                or not self._selection_complete_path.exists()
            ):
                return None
            return len(self._selected_done_items())
        return self._root_total


def _make_dep_proxies(dep_specs: Sequence[_NodeProxySpec]) -> list[_NodeStatusProxy]:
    return [_NodeStatusProxy(spec) for spec in dep_specs]


def _run_stage_in_child(
    factory: Callable[..., StageNode],
    name: str,
    store_dir: str,
    dep_specs: Sequence[_NodeProxySpec],
    factory_args: Sequence[Any],
    factory_kwargs: Mapping[str, Any],
    fail_fast: bool,
) -> dict[str, int]:
    deps = _make_dep_proxies(dep_specs)
    node = factory(
        name,
        Path(store_dir),
        deps,
        *factory_args,
        **dict(factory_kwargs),
    )
    node.run(fail_fast=fail_fast)
    return node.progress.stats()


class SubprocessStageRunNode(StageNode):
    """
    A StageNode shell whose real implementation runs in one child process.

    This wrapper is intended for stages that import JAX in ``__init__`` or
    ``process`` but should not initialize CUDA in the parent pipeline process.
    The wrapped node is constructed inside the child process.
    """

    def __init__(
        self,
        name: str,
        store_dir: Path,
        deps: list[StageNode],
        factory: Callable[..., StageNode],
        factory_args: Sequence[Any] = (),
        factory_kwargs: Mapping[str, Any] | None = None,
        child_env: Mapping[str, str | None] | None = None,
        root_total: int | None = None,
        selected_ids_store_relpath: str | None = None,
        selection_complete_relpath: str | None = None,
    ) -> None:
        super().__init__(name, store_dir, deps=deps)
        self._factory = factory
        self._factory_args = tuple(factory_args)
        self._factory_kwargs = dict(factory_kwargs or {})
        self._child_env = dict(child_env or {})
        self._root_total = root_total
        self._selected_ids_store_relpath = selected_ids_store_relpath
        self._selection_complete_relpath = selection_complete_relpath

    def _proxy_spec(self) -> _NodeProxySpec:
        dep_specs = tuple(
            dep._proxy_spec()  # type: ignore[attr-defined]
            if hasattr(dep, "_proxy_spec")
            else _NodeProxySpec(dep.name, os.fspath(dep.store_dir))
            for dep in self.deps
        )
        return _NodeProxySpec(
            name=self.name,
            store_dir=os.fspath(self.store_dir),
            deps=dep_specs,
            root_total=self._root_total,
            selected_ids_store_relpath=self._selected_ids_store_relpath,
            selection_complete_relpath=self._selection_complete_relpath,
        )

    def progress_total(self) -> int | None:
        return _NodeStatusProxy(self._proxy_spec()).progress_total()

    def ready_items(self) -> list[str]:
        return _NodeStatusProxy(self._proxy_spec()).ready_items()

    def is_complete(self) -> bool:
        return _NodeStatusProxy(self._proxy_spec()).is_complete()

    def process(self, item_id: str) -> None:  # noqa: ARG002
        raise NotImplementedError("SubprocessStageRunNode delegates run().")

    def run(
        self,
        fail_fast: bool = False,
    ) -> None:
        run_single_subprocess(
            _run_stage_in_child,
            args=(
                self._factory,
                self.name,
                os.fspath(self.store_dir),
                tuple(
                    dep._proxy_spec()
                    if hasattr(dep, "_proxy_spec")
                    else _NodeProxySpec(dep.name, os.fspath(dep.store_dir))
                    for dep in self.deps
                ),  # type: ignore[attr-defined]
                self._factory_args,
                self._factory_kwargs,
                fail_fast,
            ),
            env=self._child_env,
        )
