from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union

from .alg_base import Algorithm
from .timer import Timer


def decorate(obj: Any, decorators: List[Callable[[Any], None]]) -> None:
    for decorator in decorators:
        decorator(obj)


class TimeIt:
    def __init__(self, timer: Timer) -> None:
        self._timer = timer

    def __call__(self, alg: Algorithm) -> None:
        if "timeIt" in alg.decorators:
            return

        def _start(population: List[dict], env: Dict[str, Any]) -> None:
            alg.env["timer"] = self._timer
            self._timer.startGlobal()

        def _finish(population: List[dict], env: Dict[str, Any]) -> None:
            self._timer.stopGlobal()

        alg.addProcedure("start", _start)
        alg.addProcedure("finish", _finish)
        alg.decorators.append("timeIt")


class RankIt:
    def __call__(self, alg: Algorithm) -> None:
        if "rankIt" in alg.decorators:
            return

        def _start(population: List[dict], env: Dict[str, Any]) -> None:
            for i, _ in enumerate(population):
                population[i]["_rank"] = i

        def _enter(population: List[dict], env: Dict[str, Any]) -> None:
            solutionValueLabel = env["solutionValueLabel"]
            ranks = list(map(lambda x: x["_rank"], population))
            ranks.sort(
                key=lambda x: population[x][solutionValueLabel],
                reverse=(env["goal"] == "max"),
            )
            for i, _ in enumerate(population):
                population[ranks[i]]["_rank"] = i

        alg.addProcedure("start", _start)
        alg.addProcedure("enter", _enter)
        alg.decorators.append("rankIt")


class AddElite:
    def __init__(self, size_or_ratio: Union[int, float] = 1) -> None:
        self._size: Optional[int] = None
        self._ratio: Optional[float] = None
        if isinstance(size_or_ratio, float):
            if size_or_ratio < 0.0 and size_or_ratio > 0.0:
                raise ValueError(
                    "invalid ratio is provided, it should be within [0, 1]"
                )
            self._ratio = size_or_ratio
        elif isinstance(size_or_ratio, int):
            if size_or_ratio < 0:
                raise ValueError("invalid size is provided, it shouldn't be negative")
            self._size = size_or_ratio
        else:
            raise TypeError("expected int or float")

    def __call__(self, alg: Algorithm) -> None:
        if "addElite" in alg.decorators:
            return

        def _enter(population: List[dict], env: Dict[str, Any]) -> None:
            solutionValueLabel = env["solutionValueLabel"]
            alg.elite = {}
            pop_size = len(population)
            if self._size is None:
                assert self._ratio is not None
                self._size = int(self._ratio * pop_size)
            indexes = list(range(pop_size))
            goal = env["goal"]
            for i in range(self._size):
                for j in range(i + 1, pop_size):
                    if goal.isBetter(
                        population[indexes[j]][solutionValueLabel],
                        population[indexes[i]][solutionValueLabel],
                    ):
                        indexes[i], indexes[j] = indexes[j], indexes[i]
                alg.elite[indexes[i]] = deepcopy(population[indexes[i]])

        def _exit(population: List[dict], env: Dict[str, Any]) -> None:
            solutionValueLabel = env["solutionValueLabel"]
            for idx, elite_ind in alg.elite.items():
                if env["goal"].isBetter(
                    elite_ind[solutionValueLabel], population[idx][solutionValueLabel]
                ):
                    population[idx] = deepcopy(elite_ind)

        alg.addProcedure("enter", _enter)
        alg.addProcedure("exit", _exit)
        alg.decorators.append("addElite")
