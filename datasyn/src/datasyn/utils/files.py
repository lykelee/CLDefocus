import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Callable,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)


def makedirs_and_chmod(name: os.PathLike, mode=0o777, exist_ok=False):
    """
    Creates a directory at the specified path and explicitly sets the permissions.
    This approach ensures that the permissions are set as expected,
    which addresses potential issues on some Linux systems where os.makedirs() may not correctly apply the permissions.
    """
    # This code is a modified version of `os.makedirs`.

    head, tail = os.path.split(name)
    if not tail:
        head, tail = os.path.split(head)
    if head and tail and not os.path.exists(head):
        try:
            makedirs_and_chmod(head, mode=mode, exist_ok=exist_ok)
        except FileExistsError:
            # Defeats race condition when another thread created the path
            pass
        cdir = os.curdir
        if isinstance(tail, bytes):
            cdir = bytes(os.curdir, "ASCII")
        if tail == cdir:  # xxx/newdir/. exists if xxx/newdir exists
            return
    try:
        os.mkdir(name, mode)
        os.chmod(name, mode)
    except OSError:
        # Cannot rely on checking for EEXIST, since the operating system
        # could give priority to other errors like EACCES or EROFS
        if not exist_ok or not os.path.isdir(name):
            raise


def iter_path_pairs(
    old_root: Path, new_root: Path, pattern="*"
) -> Iterator[Tuple[Path, Path]]:
    """
    Iterate over all files under old_root and yield tuples of
    (original_path, new_path) where new_path is constructed by
    replacing the old_root with new_root.

    This would be useful to perform some conversion to all files in a root directory and save new files preserving the file structure.
    (e.g., Consider you want to make a resized version of a hierarchical image dataset.)
    """
    old_root = Path(old_root)
    new_root = Path(new_root)

    for path in old_root.rglob(pattern):
        if path.is_file():
            new_path = new_root / path.relative_to(old_root)
            yield path, new_path


class IterFilesOut(NamedTuple):
    path: Path
    idx_in_dir: int


def iter_files(
    src_root: Path,
    pred: Optional[Callable[[str], bool]] = None,
    exts: Optional[Iterable[str]] = None,
    followlinks: bool = False,
    skip_hidden: bool = True,
    sort: bool = False,
    offset: int = 0,
    limit: Optional[int] = None,
) -> Iterator[IterFilesOut]:
    """Yield all matching file paths under src_root."""
    src_root = Path(src_root)

    if pred is None:
        pred = lambda x: True

    if exts is not None:
        exts = {("." + e.lower().lstrip(".")) for e in exts}

    limit = -1 if limit is None else int(limit)
    count = 0
    for dirpath, dirnames, filenames in os.walk(src_root, followlinks=followlinks):
        if limit >= 0 and count >= limit:
            break

        dirpath = Path(dirpath)
        if skip_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            filenames = [f for f in filenames if not f.startswith(".")]
        if sort:
            dirnames.sort()
            filenames.sort()
        for fi, fname in enumerate(filenames):
            fpath = dirpath / fname

            if not pred(fpath):
                continue

            if (exts is not None) and (not fpath.suffix.lower() in exts):
                continue

            try:
                if not fpath.is_file():
                    continue
            except PermissionError:
                continue

            if offset > 0:
                offset = offset - 1
                continue

            yield IterFilesOut(path=fpath, idx_in_dir=fi)
            count += 1

            if limit >= 0 and count >= limit:
                break


def walk_and_apply(
    src_root: Path | str,
    callback: Callable[[IterFilesOut], None],
    *,
    pred: Optional[Callable[[Path], bool]] = None,
    exts: Optional[Iterable[str]] = None,
    followlinks: bool = False,
    skip_hidden: bool = True,
    limit: Optional[int] = None,
    sort: bool = True,
    aware_of_count: bool = False,
    loop_wrapper: Callable[[Iterable], Iterable] = None,
):
    src_root = Path(src_root).resolve()

    def _create_iter():
        return iter_files(
            src_root,
            pred=pred,
            exts=exts,
            followlinks=followlinks,
            skip_hidden=skip_hidden,
            sort=sort,
            limit=limit,
        )

    loop = _create_iter()

    if aware_of_count:
        from datasyn.utils.collection import IterableWithLen, count_iterable

        loop = IterableWithLen(loop, length=count_iterable(_create_iter()))

    if loop_wrapper:
        loop = loop_wrapper(loop)

    for out in loop:
        callback(out)


def convert_all_png_to_jpg(src_root: Path, dst_root: Path):
    """
    TODO: This is an example! Move this to demo code!
    """

    def step(src_path: Path):
        """
        Example callback: convert a PNG to JPG in a mirrored folder under dst_root.
        """
        from PIL import Image

        # Build destination path: same rel dir, with .jpg suffix
        rel_path = src_path.relative_to(src_root)
        out_rel = rel_path.with_suffix(".jpg")
        out_path = dst_root / out_rel
        makedirs_and_chmod(out_path.parent, exist_ok=True)

        with Image.open(src_path) as im:
            # Convert to RGB to avoid issues with PNG alpha when saving to JPG
            rgb = im.convert("RGB")
            rgb.save(out_path, format="JPEG", quality=95)

    return walk_and_apply(src_root, step, exts=[".png"])


def iter_files_with_exts(root: os.PathLike, exts: Sequence[str]):
    exts = tuple(exts)

    for r, d, files in os.walk(root):
        for file in files:
            if file.endswith(tuple(exts)):
                yield os.path.join(r, file)


def read_file_string(file: os.PathLike, encoding: str | None = None) -> str:
    """One-liner to read a file as a string."""
    with open(file, "r", encoding=encoding) as f:
        return f.read()


def write_file_string(file: os.PathLike, s: str, encoding: str | None = None) -> str:
    """One-liner to write a file as a string."""
    with open(file, "w", encoding=encoding) as f:
        return f.write(s)


def read_file_string_auto(
    path: os.PathLike,
    *,
    errors: str = "strict",
    fallback_encodings: Tuple[str, ...] = (),
    use_charset_normalizer: bool = True,
) -> Tuple[str, str]:
    """
    Reads entire file as text with automatic encoding handling.
    Returns (text, encoding_used).
    """
    import codecs

    data = Path(path).read_bytes()

    # BOMs
    if data.startswith(codecs.BOM_UTF8):
        return data[len(codecs.BOM_UTF8) :].decode("utf-8"), "utf-8-sig"
    if data.startswith(codecs.BOM_UTF32_LE) or data.startswith(codecs.BOM_UTF32_BE):
        return data.decode("utf-32"), "utf-32"
    if data.startswith(codecs.BOM_UTF16_LE) or data.startswith(codecs.BOM_UTF16_BE):
        return data.decode("utf-16"), "utf-16"

    # No BOM
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass

    if use_charset_normalizer:
        try:
            from charset_normalizer import from_bytes  # type: ignore

            res = from_bytes(data).best()
            if res and res.encoding:
                return str(res), res.encoding
        except Exception:
            pass

    for enc in fallback_encodings or ("cp949", "cp1252", "latin-1"):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue

    # Last resort: lossy but never fails
    return data.decode("latin-1", errors="replace"), "latin-1+replace"


@dataclass
class DirStats:
    """Aggregated statistics for a directory tree."""

    size: int = 0  # Total size in bytes
    files: int = 0  # Number of regular files
    dirs: int = 0  # Number of directories (excluding root)
    errors: int = 0  # Number of entries we couldn't stat/scan


def get_dir_stats(
    root: str,
    workers: int = 8,
    follow_symlinks: bool = False,
) -> DirStats:
    """
    Recursively compute directory statistics using a multithreaded scanner.

    TODO: What about `pathlib`????????????????????????

    Parameters
    ----------
    root : str
        Root directory path.
    workers : int
        Number of worker threads to use (I/O bound, so threads work well).
    follow_symlinks : bool
        Whether to follow directory symlinks. Be careful of cycles.

    Returns
    -------
    DirStats
        Aggregated size and counts.
    """
    from queue import Queue

    root = os.path.abspath(root)
    stats = DirStats()
    q: Queue[str] = Queue()
    lock = threading.Lock()

    q.put(root)

    def worker() -> None:
        nonlocal stats
        while True:
            path = q.get()
            if path is None:  # sentinel
                q.task_done()
                return
            try:
                try:
                    with os.scandir(path) as it:
                        for entry in it:
                            try:
                                if entry.is_file(follow_symlinks=follow_symlinks):
                                    st = entry.stat(follow_symlinks=follow_symlinks)
                                    with lock:
                                        stats.size += st.st_size
                                        stats.files += 1
                                elif entry.is_dir(follow_symlinks=follow_symlinks):
                                    with lock:
                                        stats.dirs += 1
                                    q.put(entry.path)
                            except (FileNotFoundError, PermissionError, OSError):
                                # Entry disappeared or cannot be accessed
                                with lock:
                                    stats.errors += 1
                except (FileNotFoundError, PermissionError, OSError):
                    # Directory itself cannot be scanned
                    with lock:
                        stats.errors += 1
            finally:
                q.task_done()

    threads = [
        threading.Thread(target=worker, daemon=True) for _ in range(max(1, workers))
    ]
    for t in threads:
        t.start()

    # Wait until all directories processed
    q.join()

    # Stop workers
    for _ in threads:
        q.put(None)
    q.join()  # ensure all sentinels processed

    return stats


def get_dir_size(root: str, **kwargs) -> int:
    """
    Convenience wrapper to get just the total size in bytes.
    Extra kwargs are passed to dir_stats (e.g., workers=16).
    """
    return get_dir_stats(root, **kwargs).size


# --- from datasyn.utils.files ---
import os
from pathlib import Path
from typing import Callable, Optional, Sequence, TypeVar

import jax.random as jr
import numpy as np

from datasyn.jaxutils.random import HRng, tag_seed
from datasyn.jaxutils.typing import *
from datasyn.utils.parallel.easy_tqdm import easy_tqdm
from datasyn.utils.parallel.mt import run_thread_pools
from datasyn.utils.time import easy_timer

T = TypeVar("T")


def ensure_directory(path: os.PathLike, mode=0o777):
    """
    Ensures there is a directory with no files and subdirectories.
    """
    from datasyn.utils.files import makedirs_and_chmod

    makedirs_and_chmod(path, mode=mode, exist_ok=True)


def ensure_clean_directory(path: os.PathLike, mode=0o777):
    """
    Ensures there is a directory with no files and subdirectories.
    """
    import shutil

    from datasyn.utils.files import makedirs_and_chmod

    path = Path(path)
    makedirs_and_chmod(path, mode=mode, exist_ok=True)
    if not path.is_dir():
        raise NotADirectoryError(path)

    for child in path.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        else:
            shutil.rmtree(child)


def create_symlink(
    file: os.PathLike,
    link: os.PathLike,
    *,
    relative: bool = False,
    overwrite: bool = False,
):
    """
    TODO: Move to my native package!

    Creates a symbolic link at `link` that points to `file`.
    """
    try:
        target = Path(file).resolve()
        link = Path(link)  # NOTE: Don't use `resolve`!

        isdir = os.path.isdir(file)

        if relative:
            target = os.path.relpath(target, link.parent)

        ensure_directory(link.parent)

        if overwrite:
            if os.path.exists(link):
                os.unlink(link)

        os.symlink(target, link, target_is_directory=isdir)

    except FileExistsError:
        raise FileExistsError(f"Error: Link already exists at {link}")
    # except FileNotFoundError:
    #    raise FileNotFoundError(f"Error: Target file not found at {file}")


def find_all_files(
    root: os.PathLike, pattern: str, predicate: Optional[Callable[[Path], bool]] = None
):
    """
    :param pattern: glob pattern
    """

    def cond(p: Path):
        if not p.is_file():
            return False
        if not (predicate is None or predicate(p)):
            return False
        return True

    root = Path(root)
    return (p for p in root.rglob(pattern) if cond(p))


def create_symlink_pairs_of_samenames(
    files_src_input: Sequence[os.PathLike],
    files_src_target: Sequence[os.PathLike],
    dir_dst_input: os.PathLike,
    dir_dst_target: os.PathLike,
):
    """
    Some paired datasets have pairs with different names (e.g., DPDD).
    Some programs work only if two files in a pair have the same name.
    If we directly put such datasets to such programs, they will be broken.
    To address this, this makes symlinks with name alignment.
    This pairs given two file sequences and assigns indices to their names (same for each pair).
    """
    from tqdm.auto import tqdm

    dir_dst_input = Path(dir_dst_input)
    dir_dst_target = Path(dir_dst_target)

    assert len(files_src_input) == len(files_src_target)
    n_total = len(files_src_input)
    n_total_digits = len(str(n_total))

    ensure_clean_directory(dir_dst_input)
    ensure_clean_directory(dir_dst_target)

    for i in tqdm(range(n_total)):
        dst_stem = str(i).zfill(n_total_digits)
        file_src_input = Path(files_src_input[i])
        file_src_target = Path(files_src_target[i])

        link_input = dir_dst_input / file_src_input.with_stem(dst_stem).name
        link_target = dir_dst_target / file_src_target.with_stem(dst_stem).name

        create_symlink(file_src_input, link_input, relative=True)
        create_symlink(file_src_target, link_target, relative=True)


def get_shuffle_plan(rng: RngKey, paths: List[T]) -> List[Tuple[T, T]]:
    """
    Generates the shuffle map using JAX's highly optimized permutation function.
    """
    n = len(paths)
    shuffled_indices = jr.permutation(rng, n).tolist()
    new_paths = [paths[i] for i in shuffled_indices]
    return list(zip(paths, new_paths))


def execute_safe_shuffle(
    files: Optional[List[Path]] = None,
    rng: Optional[RngKey] = None,
    shuffle_plan: Optional[List[Tuple[Path, Path]]] = None,
    temp_suffix: str = ".temp_shuffle",
):
    """
    Shuffles the given file names.
    This is usually meaning only if all files form a commutative collection (e.g., image datasets).

    If you want to apply the same shuffle to multiple collections, using shuffle plan might be better.
    (For example, shuffling the paired datasets.)

    TODO: Parallelization!
    """
    if not shuffle_plan:
        if not files:
            raise ValueError("Neither files and shuffle plan are given!")
        if rng is None:
            raise ValueError("`rng` is not given!")

        shuffle_plan = get_shuffle_plan(rng, files)

    temp_map: List[Tuple[Path, str]] = []

    for path_org, path_new in easy_tqdm(shuffle_plan, desc="generate temp"):
        path_temp = path_org.with_name(path_org.name + temp_suffix)
        path_org.rename(path_temp)
        temp_map.append((path_temp, path_new))

    for path_temp, path_new in easy_tqdm(temp_map, desc="rename"):
        path_temp.rename(path_new)

    print("\nShuffle complete.")


def get_files_by_names(
    root: os.PathLike,
    pred: Optional[Callable[[Path], bool]] = None,
    remove_exts: bool = False,
    fn_name: Optional[Callable[[Path], str]] = None,
    limit: Optional[int] = None,
    loop_wrapper: Callable[[Iterable], Iterable] = None,
):
    """
    Works only if all files have distinct names.

    When is it useful?
    - We have attributes of objects separated into multiple files named their own IDs.
      They have different directory structures.
    """
    from datasyn.utils.files import IterFilesOut, walk_and_apply

    names: Dict[str, Path] = dict()

    def step(out: IterFilesOut):
        file = out.path

        if fn_name is None:
            name = file.stem if remove_exts else file.name
        else:
            name = fn_name(file)

        if name in names:
            file_existing = names[name]
            raise Exception(f"Duplicated file names: {file_existing} and {file}")

        names[name] = file

    walk_and_apply(root, step, pred=pred, limit=limit, loop_wrapper=loop_wrapper)

    return names
