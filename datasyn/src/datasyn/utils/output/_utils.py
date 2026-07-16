"""Internal utilities for output management."""

import os
import shutil
from collections.abc import Iterable
from pathlib import Path

# Type alias for path-like inputs
PathLike = str | os.PathLike[str]


def normalize_paths(paths: Iterable[PathLike]) -> set[Path]:
    """Normalize an iterable of path-like objects to a set of Path.

    Args:
        paths: Iterable of str, Path, or any os.PathLike.

    Returns:
        Set of Path objects.
    """
    return {Path(p) for p in paths}


def walk_files(root: Path) -> set[Path]:
    """Return all file paths relative to root.

    Args:
        root: Directory to walk.

    Returns:
        Set of relative paths for all files under root.
    """
    if not root.exists():
        return set()

    result: set[Path] = set()
    for dirpath, _, filenames in os.walk(root):
        dir_path = Path(dirpath)
        for filename in filenames:
            result.add((dir_path / filename).relative_to(root))
    return result


def prune_empty_dirs(root: Path) -> int:
    """Remove empty directories bottom-up.

    Args:
        root: Directory to prune. The root itself is not removed.

    Returns:
        Number of directories removed.
    """
    if not root.exists():
        return 0

    removed = 0
    # Walk bottom-up by sorting paths by depth (deepest first)
    dirs: list[Path] = []
    for dirpath, dirnames, _ in os.walk(root):
        for dirname in dirnames:
            dirs.append(Path(dirpath) / dirname)

    # Sort by depth descending (deepest first)
    dirs.sort(key=lambda p: len(p.parts), reverse=True)

    for dir_path in dirs:
        try:
            if dir_path.exists() and not any(dir_path.iterdir()):
                dir_path.rmdir()
                removed += 1
        except OSError:
            # Directory not empty or permission error
            pass

    return removed


def delete_files(root: Path, relative_paths: set[Path]) -> int:
    """Delete files at specified relative paths.

    Args:
        root: Base directory.
        relative_paths: Set of relative paths to delete.

    Returns:
        Number of files deleted.
    """
    deleted = 0
    for rel_path in relative_paths:
        full_path = root / rel_path
        try:
            if full_path.is_file():
                full_path.unlink()
                deleted += 1
        except OSError:
            pass
    return deleted


def atomic_symlink(target: Path, link: Path) -> None:
    """Atomically create or replace a symlink.

    Uses the temp-file-then-rename pattern for atomicity.

    Args:
        target: The path the symlink should point to.
        link: The path where the symlink should be created.
    """
    # Create temp symlink next to the final location
    temp_link = link.with_suffix(".tmp")

    # Remove any existing temp link
    try:
        temp_link.unlink()
    except FileNotFoundError:
        pass

    # Create new symlink
    temp_link.symlink_to(target)

    # Atomically replace
    temp_link.rename(link)


def remove_tree(path: Path) -> None:
    """Remove a directory tree, handling symlinks correctly.

    Args:
        path: Directory to remove.
    """
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
