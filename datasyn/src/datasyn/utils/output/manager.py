"""Output manager facade."""

import os
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from .protocol import OutputStrategy

# Type alias for path-like inputs
PathLike = str | os.PathLike[str]


class OutputManager:
    """Facade providing convenient access to output strategies.

    Wraps an OutputStrategy with a context manager interface and
    state tracking.

    Example:
        >>> strategy = OverwriteStrategy(Path("./outputs"))
        >>> manager = OutputManager(strategy)
        >>> expected = {Path(f"chunk_{i:04d}.bin") for i in range(100)}
        >>> with manager.run(expected) as out_dir:
        ...     for i in range(100):
        ...         (out_dir / f"chunk_{i:04d}.bin").write_bytes(b"data")
    """

    def __init__(self, strategy: "OutputStrategy") -> None:
        """Initialize the manager.

        Args:
            strategy: The output strategy to use.
        """
        self._strategy = strategy
        self._output_dir: Path | None = None
        self._in_run = False

    @property
    def root(self) -> Path:
        """The root directory managed by the strategy."""
        return self._strategy.root

    @property
    def output_dir(self) -> Path:
        """Current output directory. Only valid during a run."""
        if self._output_dir is None:
            raise RuntimeError("output_dir is only valid during a run")
        return self._output_dir

    @property
    def in_run(self) -> bool:
        """Whether a run is currently in progress."""
        return self._in_run

    def begin_run(self, expected_paths: Iterable[PathLike] | None = None) -> Path:
        """Begin a new run.

        Args:
            expected_paths: Optional iterable of relative paths (str or Path)
                that will be generated.

        Returns:
            The directory where outputs should be written.

        Raises:
            RuntimeError: If a run is already in progress.
        """
        if self._in_run:
            raise RuntimeError("A run is already in progress")

        self._output_dir = self._strategy.begin_run(expected_paths)
        self._in_run = True
        return self._output_dir

    def finalize_run(self) -> None:
        """Finalize the current run.

        Raises:
            RuntimeError: If no run is in progress.
        """
        if not self._in_run:
            raise RuntimeError("No run in progress")

        self._strategy.finalize_run()
        self._in_run = False
        self._output_dir = None

    def abort_run(self) -> None:
        """Abort the current run.

        Raises:
            RuntimeError: If no run is in progress.
        """
        if not self._in_run:
            raise RuntimeError("No run in progress")

        self._strategy.abort_run()
        self._in_run = False
        self._output_dir = None

    @contextmanager
    def run(self, expected_paths: Iterable[PathLike] | None = None) -> Iterator[Path]:
        """Context manager for a complete run.

        Automatically calls begin_run on entry and finalize_run on successful
        exit. If an exception occurs, abort_run is called instead.

        Args:
            expected_paths: Optional iterable of relative paths (str or Path)
                that will be generated.

        Yields:
            The directory where outputs should be written.

        Example:
            >>> with manager.run(expected_paths) as out_dir:
            ...     generate_outputs(out_dir)
        """
        out_dir = self.begin_run(expected_paths)
        try:
            yield out_dir
            self.finalize_run()
        except BaseException:
            self.abort_run()
            raise
