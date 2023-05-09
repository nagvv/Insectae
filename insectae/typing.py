from typing import Any, Callable, Dict, TypedDict, TypeVar, Union


_T = TypeVar("_T")
Evaluable = Union[_T, Callable[..., _T]]
Individual = Dict[str, Any]

Environment = Dict[str, Any]

# TODO, + circular dependency issue
# class Environment(TypedDict, total=False):
#     timer: Timer
#     time: int
#     goal: Goal
#     target: Target
#     solutionValueLabel: str
#     solutionLabel: str
