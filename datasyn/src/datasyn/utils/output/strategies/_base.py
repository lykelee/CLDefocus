"""Base class for output strategies."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from .._utils import PathLike


class BaseStrategy(ABC):
    """Abstract base class providing common strategy functionality.

    Subclasses must implement the core lifecycle methods.
    """

    def __init__(self, root: Path) -> None:
        """Initialize the strategy.

        Args:
            root: The root directory to manage.
        """
        self._root = Path(root)
        self._current_run_dir: Path | None = None

    @property
    def root(self) -> Path:
        """The root directory managed by this strategy."""
        return self._root

    @abstractmethod
    def begin_run(self, expected_paths: Iterable[PathLike] | None = None) -> Path:
        """Prepare for a new run."""
        ...

    @abstractmethod
    def finalize_run(self) -> None:
        """Commit the run."""
        ...

    @abstractmethod
    def abort_run(self) -> None:
        """Abort the run."""
        ...

    def _ensure_root_exists(self) -> None:
        """Create the root directory if it doesn't exist."""
        self._root.mkdir(parents=True, exist_ok=True)
