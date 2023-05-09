from typing import Any, Dict

from .metrics import Metrics
from .typing import Environment


class StopMaxGeneration:
    def __init__(self, maxGen: int, metrics: Metrics) -> None:
        self._metrics = metrics
        self._maxGen = maxGen

    def __call__(self, env: Environment) -> bool:
        self._metrics.newGeneration()  # TODO move out from here, leave only checks
        return self._metrics.currentGeneration > self._maxGen


class StopMaxEF:
    def __init__(self, maxEF: int, metrics: Metrics) -> None:
        self._metrics = metrics
        self._maxEF = maxEF

    def __call__(self, env: Environment) -> bool:
        self._metrics.newGeneration()
        return self._metrics.efs > self._maxEF


class StopValue:
    def __init__(self, value: Any, maxGen: int, metrics: Metrics) -> None:
        self._metrics = metrics
        self._value = value
        self._maxGen = maxGen

    def __call__(self, env: Environment) -> bool:
        self._metrics.newGeneration()
        goal = env["goal"]
        valIsBetter = goal.isBetter(self._value, self._metrics.bestValue)
        genExceeded = self._metrics.currentGeneration >= self._maxGen
        return not valIsBetter or genExceeded
