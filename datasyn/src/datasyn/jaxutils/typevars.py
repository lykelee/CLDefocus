from datasyn.jaxutils.typing import *

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")
F = TypeVar("F", bound=Callable)

T_PyTree = TypeVar("T_PyTree", bound=PyTree)
