from typing import Any, Callable, Dict, List, Mapping, TypeVar, Union

import numpy as np

# from typing import TYPE_CHECKING

# if TYPE_CHECKING:  # avoid circular dependency issue
#     from .timer import Timer
#     from .goals import Goal
#     from .targets import Target

_T = TypeVar("_T")
Individual = Dict[str, Any]
FuncKWArgs = Dict[str, Any]
Environment = Dict[str, Any]
Evaluable = Union[_T, Callable[[int, np.random.Generator], _T]]


# class Environment(TypedDict, total=False):
#     timer: 'Timer'
#     time: int
#     goal: 'Goal'
#     target: 'Target'
#     solutionValueLabel: str
#     solutionLabel: str
