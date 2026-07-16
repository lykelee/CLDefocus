"""Output strategy protocol definition."""

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

# Type alias for path-like inputs
PathLike = str | os.PathLike[str]


class OutputStrategy(Protocol):
    """Pluggable strategy for output management.

    Implementations handle the lifecycle of output directories across runs,
    with different policies for cleanup, retention, and atomicity.
    """

    @property
    def root(self) -> Path:
        """The root directory managed by this strategy."""
        ...

    def begin_run(self, expected_paths: Iterable[PathLike] | None = None) -> Path:
        """Prepare for a new run.

        Args:
            expected_paths: If provided, iterable of relative paths (str or Path)
                that will be generated. Enables set-based cleanup optimization.

        Returns:
            The directory where outputs should be written.
        """
        ...

    def finalize_run(self) -> None:
        """Commit the run. Called after generation completes successfully.

        Performs cleanup, rotation, or archival depending on strategy.
        """
        ...

    def abort_run(self) -> None:
        """Abort the run. Called if generation fails.

        Strategy decides whether to keep partial outputs.
        """
        ...
