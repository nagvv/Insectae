from typing import Any


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
