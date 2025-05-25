from copy import deepcopy
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .alg_base import Algorithm
from .common import evalf, l2metrics
from .helpers import wrap_executor_evaluate
from .operators import copyAttribute
from .targets import Target
from .timer import Timer
from .typing import Environment, Evaluable, Individual


def decorate(obj: Any, decorators: List[Callable[[Any], None]]) -> None:
    for decorator in decorators:
        decorator(obj)


class TimeIt:
    def __init__(self, timer: Timer) -> None:
        self._timer = timer

    def __call__(self, alg: Algorithm) -> None:
        if "timeIt" in alg.decorators:
            return

        alg.env["timer"] = self._timer
        alg.addProcedure("start", self._start)
        alg.addProcedure("finish", self._finish)
        alg.decorators.append("timeIt")

    @staticmethod
    def _start(population: List[Individual], env: Environment) -> None:
        env["timer"].startGlobal()

    @staticmethod
    def _finish(population: List[Individual], env: Environment) -> None:
        env["timer"].stopGlobal()


class RankIt:
    def __call__(self, alg: Algorithm) -> None:
        if "rankIt" in alg.decorators:
            return

        alg.addProcedure("start", self._start)
        alg.addProcedure("enter", self._enter)
        alg.decorators.append("rankIt")

    @staticmethod
    def _start(population: List[Individual], env: Environment) -> None:
        for i, _ in enumerate(population):
            population[i]["_rank"] = i

    @staticmethod
    def _enter(population: List[Individual], env: Environment) -> None:
        solutionValueLabel = env["solutionValueLabel"]
        ranks = list(map(lambda x: x["_rank"], population))
        ranks.sort(
            key=lambda x: population[x][solutionValueLabel],
            reverse=(env["goal"] == "max"),
        )
        for i, _ in enumerate(population):
            population[ranks[i]]["_rank"] = i


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

        alg.addProcedure("enter", self._enter)
        alg.addProcedure("exit", self._exit)
        alg.decorators.append("addElite")

    def _enter(self, population: List[Individual], env: Environment) -> None:
        solutionValueLabel = env["solutionValueLabel"]
        elite = env["elite"] = {}
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
            elite[indexes[i]] = deepcopy(population[indexes[i]])

    @staticmethod
    def _exit(population: List[dict], env: Environment) -> None:
        solutionValueLabel = env["solutionValueLabel"]
        for idx, elite_ind in env["elite"].items():
            if env["goal"].isBetter(
                elite_ind[solutionValueLabel], population[idx][solutionValueLabel]
            ):
                population[idx] = deepcopy(elite_ind)


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

    NICHE_COUNT_KEY = "_fitness_sharing_niche_count"
    ORIG_F_KEY = "_fitness_sharing_orig_f"

    def __init__(
        self,
        sigma: float,
        alpha: float = 1.0,
        dist_func: Callable[[NDArray, NDArray], float] = l2metrics,
        beta: Optional[Evaluable[float]] = None,
        niched_solution_value_key: str = "fitness_sharing_f",
    ) -> None:
        self._sigma = sigma
        self._alpha = alpha
        self._dist_func = dist_func
        self._beta = beta
        self._niched_solution_value_key = niched_solution_value_key
        self._env = None
        self._restore_evaluate = None

    def __call__(self, algorithm: Algorithm) -> None:
        if "AddFitnessSharing" in algorithm.decorators:
            return

        algorithm.addProcedure("start", self._start)
        algorithm.addProcedure("finish", self._finish)
        algorithm.decorators.append("AddFitnessSharing")
        self._env = algorithm.env

    @staticmethod
    def _sharing_function(
        pair, dist_func: Callable, alpha: float, sigma: float
    ) -> float:
        distance = dist_func(pair[0], pair[1])
        if distance < sigma:
            return 1.0 - (distance / sigma) ** alpha
        return 0.0

    @staticmethod
    def _post(ind: Individual, paired_values: Iterable[Tuple[float, Individual]]):
        # 1.0 from own fitness value
        ind[AddFitnessSharing.NICHE_COUNT_KEY] = 1.0 + np.sum(
            [v for v, _ in paired_values]
        )

    @staticmethod
    def _apply_niching(ind: Individual, keyf: str, beta: Optional[float], toMax: bool):
        f_value = ind[AddFitnessSharing.ORIG_F_KEY]
        if beta is not None:
            f_value **= beta
        if toMax:
            f_value /= ind[AddFitnessSharing.NICHE_COUNT_KEY]
        else:
            f_value *= ind[AddFitnessSharing.NICHE_COUNT_KEY]
        ind[keyf] = f_value

    def _evaluate(
        self,
        self_executor,
        population: List[Individual],
        keyx: str,
        keyf: str,
        target: Target,
        **kwargs,
    ) -> None:
        assert isinstance(self._env, dict)
        timer = self._env.get("timer")
        self_executor.allNeighbors(
            population=population,
            op=self._sharing_function,
            op_fnkwargs={
                "dist_func": self._dist_func,
                "alpha": self._alpha,
                "sigma": self._sigma,
            },
            op_getter=keyx,
            post=self._post,
            post_fnkwargs={},
            timingLabel="fitness_sharing",
            timer=timer,
        )
        self_executor._fitness_sharing_orig_evaluate(
            population=population,
            keyx=keyx,
            keyf=AddFitnessSharing.ORIG_F_KEY,
            target=target,
            **kwargs,
        )
        self_executor.foreach(
            population=population,
            op=self._apply_niching,
            fnkwargs={
                "keyf": keyf,
                "beta": evalf(self._beta, self._env["time"], self._env["rng"]),
                "toMax": self._env["goal"] == "max",
            },
            timingLabel="fitness_sharing",
            timer=timer,
        )

    def _start(self, _: List[Individual], env: Environment) -> None:
        executor = env["executor"]
        self._restore_evaluate = wrap_executor_evaluate(
            executor, self._evaluate, "_fitness_sharing_orig_evaluate"
        )

    def _finish(self, population: List[Individual], env: Environment) -> None:
        keyf = env["solutionValueLabel"]

        assert callable(self._restore_evaluate)
        self._restore_evaluate()
        executor = env["executor"]
        executor.foreach(
            population,
            op=copyAttribute,
            fnkwargs={
                "keyFrom": keyf,
                "keyTo": self._niched_solution_value_key,
            },
        )
        # run evaluate again without fitness sharing to get the actual
        # fitness values; we can't simply replace them with existing orig
        # values since after evaluate algorithm can tamper with them
        executor.evaluate(
            population=population,
            keyx=env["solutionLabel"],
            keyf=keyf,
            target=env["target"],
            timingLabel="fitness_sharing",
            timer=env.get("timer"),
        )


class AddClearing:
    """
    The class to inject the clearing niching mechanism into the given algorithm.
    It implements clearing procedure that is described in the paper [1]_.

    This class is applicable to any algorithm which uses evaluate pattern from
    its executor. Any other fitness evaluations outside that are not tracked.

    Parameters
    ----------
    sigma : float
        The clearing radius.
    capacity : int
        The capacity of each niche.
    dist_func : Callable[[NDArray, NDArray], float], default: l2metrics
        The distance function used to compute the distance between individuals.
    niched_solution_value_key : str, default: "clearing_f"
        The key under which the niched solution value is stored after the
        algorithm is executed.

    References
    -----
    .. [1] Pétrowski A. A clearing procedure as a niching method for genetic
       algorithms //Proceedings of IEEE international conference on evolutionary
       computation. – IEEE, 1996. – С. 798-803.
    """

    ORIG_F_KEY = "_clearing_orig_f"
    IS_NICHE_WINNER_KEY = "_clearing_is_niche_winner"

    def __init__(
        self,
        sigma: float,
        capacity: int,
        dist_func: Callable[[NDArray, NDArray], float] = l2metrics,
        niched_solution_value_key: str = "clearing_f",
    ) -> None:
        self._sigma = sigma
        self._capacity = capacity
        self._dist_func = dist_func
        self._niched_solution_value_key = niched_solution_value_key
        self._to_max = False
        self._env = None
        self._restore_evaluate = None

    def __call__(self, algorithm: Algorithm) -> None:
        if "AddClearing" in algorithm.decorators:
            return

        self._to_max = algorithm.goal == "max"
        algorithm.addProcedure("start", self._start)
        algorithm.addProcedure("finish", self._finish)
        algorithm.decorators.append("AddClearing")
        self._env = algorithm.env

    @staticmethod
    def _compute_distances(pair, dist_func: Callable[[float, float], float]) -> float:
        distance = dist_func(pair[0], pair[1])
        return distance

    @staticmethod
    def _compute_niche(
        ind,
        paired_dists: Iterable[Tuple[float, Individual]],
        sigma: float,
        capacity: int,
        to_max: bool,
    ) -> None:
        # stores the following values: (disance, fitness value, is ind)
        niche = np.fromiter(
            (
                (d, ind[AddClearing.ORIG_F_KEY], 0)
                for d, ind in paired_dists
                if d <= sigma
            ),
            dtype=np.dtype((float, 3)),
        )
        if len(niche) == 0:
            ind[AddClearing.IS_NICHE_WINNER_KEY] = True
            return

        niche = np.concatenate((niche, [(0, ind[AddClearing.ORIG_F_KEY], 1)]), axis=0)
        cap = min(capacity, len(niche))
        if to_max is True:
            niche = niche[(-niche[:, 1]).argpartition(kth=cap - 1)[:cap]]
        else:
            niche = niche[niche[:, 1].argpartition(kth=cap - 1)[:cap]]
        ind[AddClearing.IS_NICHE_WINNER_KEY] = np.count_nonzero(niche[:, 2])

    @staticmethod
    def _apply_niching(ind: Individual, keyf: str, to_max: bool) -> None:
        if ind[AddClearing.IS_NICHE_WINNER_KEY] != 0:
            ind[keyf] = ind[AddClearing.ORIG_F_KEY]
        else:
            ind[keyf] = -np.inf if to_max else np.inf

    def _evaluate(
        self,
        self_executor,
        population: List[Individual],
        keyx: str,
        keyf: str,
        target: Target,
        **kwargs,
    ) -> None:
        assert isinstance(self._env, dict)
        timer = self._env.get("timer")
        self_executor._clearing_orig_evaluate(
            population=population,
            keyx=keyx,
            keyf=AddClearing.ORIG_F_KEY,
            target=target,
            **kwargs,
        )
        self_executor.allNeighbors(
            population=population,
            op=self._compute_distances,
            op_fnkwargs={"dist_func": self._dist_func},
            op_getter=keyx,
            post=self._compute_niche,
            post_fnkwargs={
                "sigma": self._sigma,
                "capacity": self._capacity,
                "to_max": self._to_max,
            },
            timingLabel="clearing",
            timer=timer,
        )
        self_executor.foreach(
            population=population,
            op=self._apply_niching,
            fnkwargs={"keyf": keyf, "to_max": self._to_max},
            timingLabel="clearing",
            timer=timer,
        )

    def _start(self, _: List[Individual], env: Environment) -> None:
        executor = env["executor"]
        self._restore_evaluate = wrap_executor_evaluate(
            executor, self._evaluate, "_clearing_orig_evaluate"
        )

    def _finish(self, population: List[Individual], env: Environment) -> None:
        keyf = env["solutionValueLabel"]

        assert callable(self._restore_evaluate)
        self._restore_evaluate()
        executor = env["executor"]
        executor.foreach(
            population,
            op=copyAttribute,
            fnkwargs={
                "keyFrom": keyf,
                "keyTo": self._niched_solution_value_key,
            },
        )
        # run evaluate again without niching to get the actual solution
        # values; we can't simply replace them with existing orig values
        # since after evaluate algorithm can tamper with them
        executor.evaluate(
            population=population,
            keyx=env["solutionLabel"],
            keyf=keyf,
            target=env["target"],
            timingLabel="clearing",
            timer=env.get("timer"),
        )
