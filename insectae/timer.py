from time import time
from typing import Any, Callable

from .metrics import Metrics


class Timer:
    def __init__(self, metrics: Metrics) -> None:
        self.metrics = metrics
        self.timeGlobal: float = 0.0
        self.timeLocal: float = 0.0

    def startGlobal(self) -> None:
        self.timeGlobal = time()

    def stopGlobal(self) -> None:
        self.metrics.timing["total"] = time() - self.timeGlobal

    def startLocal(self) -> None:
        self.timeLocal = time()

    def stopLocal(self, label: str) -> None:
        if label not in self.metrics.timing:
            self.metrics.timing[label] = 0.0
        self.metrics.timing[label] += time() - self.timeLocal


def timing(target_func: Callable) -> Callable:
    def func(*arg, timingLabel: str = None, timer: Timer = None, **kwargs) -> Any:
        doTiming = (timingLabel is not None) and (timer is not None)
        if doTiming:
            timer.startLocal()
        result = target_func(*arg, **kwargs)
        if doTiming:
            timer.stopLocal(timingLabel)
        return result

    return func
