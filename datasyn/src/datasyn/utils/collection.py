from typing import Iterable, Iterator, List, TypeVar

T = TypeVar("T")


def list_grid(shape, fill=None) -> List:
    """
    Generates an empty multidimensional array represented as nested Python lists.
    """
    if len(shape) == 1:
        return [fill] * shape[0]
    return [list_grid(shape[1:]) for _ in range(shape[0])]


class IterableWithLen(Iterable[T]):
    def __init__(self, iterable: Iterable[T], length: int):
        self._it = iterable
        self._len = length

    def __iter__(self) -> Iterator[T]:
        return iter(self._it)

    def __len__(self) -> int:
        return self._len


def count_iterable(it: Iterable):
    """
    NOTE: For infinite iterables, this will be stuck!
    """
    if hasattr(it, "__len__"):
        return len(it)
    return sum(1 for _ in it)
