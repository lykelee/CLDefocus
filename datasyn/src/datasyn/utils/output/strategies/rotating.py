"""Rotating strategy - fresh directory per run with symlink switching."""

import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Literal

from .._utils import PathLike, atomic_symlink, remove_tree
from ._base import BaseStrategy


class RotatingStrategy(BaseStrategy):
    """Strategy that creates fresh directories for each run.

    Each run writes to a new directory. A 'current' symlink points to the
    latest completed run. Old runs are deleted based on the keep parameter.

    Directory structure:
        outputs/
        ├── current -> run_003/     (symlink to latest)
        ├── run_001/                (oldest, may be deleted)
        ├── run_002/
        └── run_003/                (latest completed run)

    Example:
        >>> strategy = RotatingStrategy(Path("./outputs"), keep=3)
        >>> out_dir = strategy.begin_run()  # Creates new run directory
        >>> # ... generate outputs to out_dir ...
        >>> strategy.finalize_run()  # Updates symlink, deletes old runs
    """

    CURRENT_LINK_NAME = "current"

    def __init__(
        self,
        root: Path,
        *,
        keep: int = 1,
        naming: Literal["sequential", "timestamp"] = "sequential",
    ) -> None:
        """Initialize the rotating strategy.

        Args:
            root: The root directory for outputs.
            keep: Number of old runs to retain (in addition to current).
                0 means keep only the current run. Defaults to 1.
            naming: How to name run directories.
                "sequential": run_001, run_002, ...
                "timestamp": run_20240115_143022, ...
                Defaults to "sequential".
        """
        super().__init__(root)
        self._keep = keep
        self._naming = naming

    @property
    def current_link(self) -> Path:
        """Path to the 'current' symlink."""
        return self._root / self.CURRENT_LINK_NAME

    def begin_run(self, expected_paths: Iterable[PathLike] | None = None) -> Path:
        """Prepare for a new run by creating a fresh directory.

        Args:
            expected_paths: Ignored for this strategy (no set-based cleanup).

        Returns:
            The new run directory where outputs should be written.
        """
        self._ensure_root_exists()

        # Generate new run directory name
        run_name = self._generate_run_name()
        run_dir = self._root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        self._current_run_dir = run_dir
        return run_dir

    def finalize_run(self) -> None:
        """Finalize the run by updating symlink and cleaning up old runs."""
        if self._current_run_dir is None:
            return

        # Atomically update the current symlink
        atomic_symlink(
            self._current_run_dir.name,  # Relative path
            self.current_link,
        )

        # Clean up old runs
        self._cleanup_old_runs()

        self._current_run_dir = None

    def abort_run(self) -> None:
        """Abort the run by deleting the incomplete run directory."""
        if self._current_run_dir is not None:
            remove_tree(self._current_run_dir)
            self._current_run_dir = None

    def _generate_run_name(self) -> str:
        """Generate a name for the new run directory."""
        if self._naming == "timestamp":
            return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:  # sequential
            return f"run_{self._next_sequential_id():03d}"

    def _next_sequential_id(self) -> int:
        """Find the next sequential ID for run directories."""
        max_id = 0
        pattern = re.compile(r"^run_(\d+)$")

        for entry in self._root.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                match = pattern.match(entry.name)
                if match:
                    max_id = max(max_id, int(match.group(1)))

        return max_id + 1

    def _get_run_dirs(self) -> list[Path]:
        """Get all run directories sorted by age (oldest first)."""
        run_dirs: list[Path] = []

        # Match both sequential and timestamp naming
        pattern = re.compile(r"^run_(\d+|\d{8}_\d{6})$")

        for entry in self._root.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                if pattern.match(entry.name):
                    run_dirs.append(entry)

        # Sort by name (works for both sequential and timestamp)
        run_dirs.sort(key=lambda p: p.name)
        return run_dirs

    def _cleanup_old_runs(self) -> None:
        """Delete old run directories exceeding the keep limit."""
        run_dirs = self._get_run_dirs()

        # Determine which directory is currently pointed to
        current_target: Path | None = None
        if self.current_link.is_symlink():
            try:
                current_target = self.current_link.resolve()
            except OSError:
                pass

        # Filter out the current directory
        old_dirs = [d for d in run_dirs if d.resolve() != current_target]

        # Delete oldest directories exceeding keep limit
        dirs_to_delete = old_dirs[: max(0, len(old_dirs) - self._keep)]
        for dir_path in dirs_to_delete:
            remove_tree(dir_path)
