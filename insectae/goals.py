from functools import cmp_to_key
from typing import Any, Callable


class Goal:
    """The base class that determines the direction of optimization."""

    _dirStr = str()

    def isBetter(self, x: Any, y: Any) -> bool:
        raise NotImplementedError

    def getDir(self) -> str:
        return self._dirStr

    def __eq__(self, other) -> bool:
        if not isinstance(other, str):
            return NotImplemented
        return other == self._dirStr

    def __str__(self) -> str:
        return self._dirStr

    def get_cmp_to_key(self, accessor: Callable[[Any], Any] = lambda x: x):
        def comparator(obj1: Any, obj2: Any):
            v1, v2 = accessor(obj1), accessor(obj2)
            if self.isBetter(v1, v2):
                return -1
            elif self.isBetter(v1, v2):
                return 1
            else:
                return 0

        return cmp_to_key(comparator)


class ToMax(Goal):
    _dirStr = "max"

    def isBetter(self, x: Any, y: Any) -> bool:
        return x > y


class ToMin(Goal):
    _dirStr = "min"

    def isBetter(self, x: Any, y: Any) -> bool:
        return x < y


def getGoal(goal: str) -> Goal:
    """An helper function to get corresponding goal class from string."""
    if goal == "min":
        return ToMin()
    elif goal == "max":
        return ToMax()
    raise ValueError("Unknown goal name")
