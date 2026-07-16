"""Overwrite strategy - set-based cleanup with in-place overwrites."""

from collections.abc import Iterable
from pathlib import Path

from .._utils import (
    PathLike,
    delete_files,
    normalize_paths,
    prune_empty_dirs,
    walk_files,
)
from ._base import BaseStrategy


class OverwriteStrategy(BaseStrategy):
    """Strategy that overwrites outputs in place with set-based cleanup.

    Before each run, deletes stale files (those not in the expected set),
    then allows the run to overwrite existing files and create new ones.

    This strategy exploits high overlap between runs by avoiding
    unlink+create for files that will be overwritten anyway.

    Example:
        >>> strategy = OverwriteStrategy(Path("./outputs"))
        >>> expected = {Path("a.txt"), Path("b.txt")}
        >>> out_dir = strategy.begin_run(expected)  # Deletes stale files
        >>> # ... generate outputs to out_dir ...
        >>> strategy.finalize_run()  # No-op for this strategy
    """

    def __init__(
        self,
        root: Path,
        *,
        cleanup_empty_dirs: bool = True,
    ) -> None:
        """Initialize the overwrite strategy.

        Args:
            root: The root directory for outputs.
            cleanup_empty_dirs: Whether to prune empty directories after
                deleting stale files. Defaults to True.
        """
        super().__init__(root)
        self._cleanup_empty_dirs = cleanup_empty_dirs

    def begin_run(self, expected_paths: Iterable[PathLike] | None = None) -> Path:
        """Prepare for a new run by cleaning up stale files.

        If expected_paths is provided, deletes files not in that set.
        If expected_paths is None, no cleanup is performed (overwrite-only mode).

        Args:
            expected_paths: Iterable of relative paths (str or Path) that will
                be generated. If None, skips cleanup and only overwrites.

        Returns:
            The root directory where outputs should be written.
        """
        self._ensure_root_exists()

        if expected_paths is not None:
            # Normalize to set[Path] for consistent comparison
            expected_normalized = normalize_paths(expected_paths)

            # Find and delete stale files
            existing = walk_files(self._root)
            stale = existing - expected_normalized
            delete_files(self._root, stale)

            # Optionally prune empty directories
            if self._cleanup_empty_dirs:
                prune_empty_dirs(self._root)

        self._current_run_dir = self._root
        return self._root

    def finalize_run(self) -> None:
        """Finalize the run. No-op for overwrite strategy."""
        self._current_run_dir = None

    def abort_run(self) -> None:
        """Abort the run. Leaves partial outputs in place."""
        self._current_run_dir = None
