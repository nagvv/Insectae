from copy import deepcopy
from types import MethodType
from typing import Any, Callable, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .alg_base import Algorithm
from .common import l2metrics
from .timer import Timer
from .typing import Environment, Individual


def decorate(obj: Any, decorators: List[Callable[[Any], None]]) -> None:
    for decorator in decorators:
        decorator(obj)


class TimeIt:
    def __init__(self, timer: Timer) -> None:
        self._timer = timer

    def __call__(self, alg: Algorithm) -> None:
        if "timeIt" in alg.decorators:
            return

        def _start(population: List[Individual], env: Environment) -> None:
            alg.env["timer"] = self._timer
            self._timer.startGlobal()

        def _finish(population: List[Individual], env: Environment) -> None:
            self._timer.stopGlobal()

        alg.addProcedure("start", _start)
        alg.addProcedure("finish", _finish)
        alg.decorators.append("timeIt")


class RankIt:
    def __call__(self, alg: Algorithm) -> None:
        if "rankIt" in alg.decorators:
            return

        def _start(population: List[Individual], env: Environment) -> None:
            for i, _ in enumerate(population):
                population[i]["_rank"] = i

        def _enter(population: List[Individual], env: Environment) -> None:
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

        def _enter(population: List[Individual], env: Environment) -> None:
            solutionValueLabel = env["solutionValueLabel"]
            # TODO use item with changeable key inside env instead of alg
            setattr(alg, "elite", {})
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
                getattr(alg, "elite")[indexes[i]] = deepcopy(population[indexes[i]])

        def _exit(population: List[dict], env: Environment) -> None:
            solutionValueLabel = env["solutionValueLabel"]
            for idx, elite_ind in getattr(alg, "elite").items():
                if env["goal"].isBetter(
                    elite_ind[solutionValueLabel], population[idx][solutionValueLabel]
                ):
                    population[idx] = deepcopy(elite_ind)

        alg.addProcedure("enter", _enter)
        alg.addProcedure("exit", _exit)
        alg.decorators.append("addElite")


class AddFitnessSharing:
    """
    The class to inject the fitness sharing niching mechanism into the given
    algorithm. The class implements fitness sharing that is described in the
    paper [1]_. For minimization problem fitness value is multiplied by the
    value of niche count.

    This class is applicable to any algorithm which uses evaluate pattern from
    its executor. Any other fitness evaluations outside that are not tracked.

    Parameters
    ----------
    sigma : float
        The dissimilarity threshold  (also called the distance cutoff or the
        niche radius)
    alpha : float, default: 1.0
        The constant parameter which regulates the shape of the sharing
        function.
    dist_func : Callable[[NDArray, NDArray], float], default: l2metrics
        The distance function used to compute the distance between individuals.
    beta : float, optional
        An optional fitness scaling parameter. The default value is None, which
        means that no scaling is applied (equivalent to 1.0). See References.
    niched_solution_value_key : str, default: "fitness_sharing_f"
        The key under which the niched solution value is stored after the
        algorithm is executed.

    References
    -----
    .. [1] Sareni B., Krahenbuhl L. Fitness sharing and niching methods
       revisited //IEEE transactions on Evolutionary Computation. – 1998.
       – Т. 2. – №. 3. – С. 97-106.
    """

    def __init__(
        self,
        sigma: float,
        alpha: float = 1.0,
        dist_func: Callable[[NDArray, NDArray], float] = l2metrics,
        beta: Optional[float] = None,
        niched_solution_value_key: str = "fitness_sharing_f",
    ) -> None:
        self._sigma = sigma
        self._alpha = alpha
        self._dist_func = dist_func
        self._beta = beta
        self._orig_f_key = "_fitness_sharing_orig_f"
        self._niche_count_key = "_fitness_sharing_niche_count"
        self._niched_solution_value_key = niched_solution_value_key

    def __call__(self, algorithm: Algorithm) -> None:
        if "AddFitnessSharing" in algorithm.decorators:
            return

        def _evaluate(
            self_executor,
            population: List[Individual],
            keyx: str,
            keyf: str,
            env: Environment,
            **kwargs
        ) -> None:
            timer = env.get("timer")
            self_executor.signals(
                population=population,
                metrics=self._dist_func,
                shape=lambda d, **kwargs: self._sharing_function(d),
                reduce=np.sum,
                keyx=keyx,
                keys=self._niche_count_key,
                env=env,
                timingLabel="fitness_sharing",
                timer=timer,
            )
            self_executor._fitness_sharing_orig_evaluate(
                population=population,
                keyx=keyx,
                keyf=self._orig_f_key,
                env=env,
                **kwargs
            )
            self_executor.foreach(
                population=population,
                op=self._apply_niching,
                fnkwargs={"keyf": keyf, "toMax": algorithm.goal == "max"},
                timingLabel="fitness_sharing",
                timer=timer,
            )

        def _start(_: List[Individual], env: Environment) -> None:
            executor = algorithm.executor
            executor._fitness_sharing_orig_evaluate = executor.evaluate
            executor.evaluate = MethodType(_evaluate, executor)

        def _finish(population: List[Individual], env: Environment) -> None:
            keyf = env["solutionValueLabel"]

            def _save_niched_solution_value(ind: Individual):
                ind[self._niched_solution_value_key] = ind[keyf]

            executor = algorithm.executor
            executor.evaluate = executor._fitness_sharing_orig_evaluate
            del executor._fitness_sharing_orig_evaluate
            executor.foreach(
                population=population, op=_save_niched_solution_value, fnkwargs={}
            )
            # run evaluate again without fitness sharing to get the actual
            # fitness values; we can't simply replace them with existing orig
            # values since after evaluate algorithm can tamper with them
            executor.evaluate(
                population=population,
                keyx=env["solutionLabel"],
                keyf=keyf,
                env=env,
                timingLabel="fitness_sharing",
                timer=env.get("timer"),
            )

        algorithm.addProcedure("start", _start)
        algorithm.addProcedure("finish", _finish)
        algorithm.decorators.append("AddFitnessSharing")

    def _sharing_function(self, distance: float) -> float:
        if distance < self._sigma:
            return 1.0 - (distance / self._sigma) ** self._alpha
        return 0.0

    def _apply_niching(self, ind: Individual, keyf: str, toMax: bool):
        f_value = ind[self._orig_f_key]
        if self._beta is not None:
            f_value **= self._beta
        if toMax:
            f_value /= ind[self._niche_count_key]
        else:
            f_value *= ind[self._niche_count_key]
        ind[keyf] = f_value
