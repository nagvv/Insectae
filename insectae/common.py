import copy
from operator import itemgetter
from typing import Any, Callable, List, TypeVar, Union
from bisect import bisect

import numpy as np
from numpy.typing import NDArray

from .goals import Goal
from .patterns import neighbors, pairs
from .timer import timing
from .typing import Environment, Evaluable, Individual

# TODO move some to the operators.py or move these here

_T = TypeVar("_T")


def evalf(param: Evaluable[_T], time: int) -> _T:
    if callable(param):
        return param(time)
    return param


class FillAttribute:
    def __init__(self, op: Any) -> None:
        self.op = op

    def __call__(self, ind: Individual, key: str, **opkwargs) -> None:
        if callable(self.op):
            ind[key] = self.op(**opkwargs)
        elif np.isscalar(self.op):
            ind[key] = self.op
        else:
            # FIXME: is there type that is not scalar and doesn't define copy()
            ind[key] = self.op.copy()


def copyAttribute(ind: Individual, keyFrom: str, keyTo: str) -> None:
    if np.isscalar(ind[keyFrom]):
        ind[keyTo] = ind[keyFrom]
    else:
        ind[keyTo] = ind[keyFrom].copy()


def weighted_choice(values: List[Any], weights: List[float], rng: np.random.Generator):
    """
    Select a random element from `values` with probabilities proportional to `weights`.

    This function is the equivalent of `choices(values, weights=weights)[0]`
    where `choices` is from `random` module. Which is equivalent to
    `rng.choice(values, p=weights/sum(weights))`.

    We can't use `choices` because we want to use seeded numpy random number
    generator, but `choice` from numpy is extremely slow compared to `choices`
    from `random` module and this function.

    This function assumes that all input parameters are valid and does not validate them.

    Parameters
    ----------
    values : List[Any]
        The list of elements to choose from.
    weights : List[float]
        The list of weights corresponding to each element in `values`.
    rng : np.random.Generator
        A NumPy random generator instance for generating random numbers.
    """

    total = 0
    cum_weights = []
    for w in weights:
        total += w
        cum_weights.append(total)
    x = rng.random() * total
    i = bisect(cum_weights, x)
    return values[i]


class Mixture:
    def __init__(self, methods: List[Callable[..., None]], probs: List[float]) -> None:
        if len(methods) != len(probs):
            raise ValueError("The lengths of `methods` and `probs` must be the same.")
        if any(w < 0 for w in probs):
            raise ValueError("`probs` must be non-negative.")
        if sum(probs) == 0:
            raise ValueError("The sum of probs must be greater than zero.")

        self.methods = methods + [self._noop]  # append no-op
        self.probs = probs + [1.0 - np.sum(probs)]

    @staticmethod
    def _noop(*args, **kwargs):
        pass

    def __call__(self, inds: List[Individual], env: Environment, **opkwargs) -> None:
        method = weighted_choice(self.methods, self.probs, env["rng"])
        method(inds, env=env, **opkwargs)


class ProbOp:
    def __init__(
        self, method: Callable[..., None], prob: Union[float, Callable[..., float]]
    ) -> None:
        self.method = method
        self.prob = prob

    # TODO: ind_or_inds is not good
    def __call__(
        self, ind_or_inds: Union[Individual, List[Individual]], env: Environment, **opkwargs
    ) -> None:
        prob = evalf(self.prob, env["time"])
        if env["rng"].random() < prob:
            self.method(ind_or_inds, env=env, **opkwargs)


class TimedOp:
    def __init__(self, method: Callable[..., None], dt: int) -> None:
        self.method = method
        self.dt = dt

    def __call__(self, ind_or_inds: Union[Individual, List[Individual]], **opkwargs) -> None:
        time = opkwargs["env"]["time"]
        if time % self.dt == 0:
            self.method(ind_or_inds, **opkwargs)


class Shuffled:
    def __init__(self, op: Callable[..., None]) -> None:
        self.op = op

    def __call__(self, population: List[Individual], env: Environment, **opkwargs) -> None:
        perm = list(range(len(population)))
        env["rng"].shuffle(perm)
        # TODO pass executor
        neighbors(population, self.op, perm, env=env, **opkwargs)


def samplex(n: int, m: int, exclude: List[int], rng: np.random.Generator) -> List[int]:
    s = list(set(range(n)) - set(exclude))
    return list(rng.choice(s, m, False))


class Selected:
    def __init__(self, op: Callable[..., None]) -> None:
        self.op = op

    def __call__(self, population: List[Individual], env: Environment, **opkwargs) -> None:
        shadow: List[Individual] = []
        rng = env["rng"]
        for i in range(len(population)):
            j = samplex(len(population), 1, [i], rng)[0]
            shadow.append(copy.deepcopy(population[j]))
        # TODO pass executor
        pairs(population, shadow, self.op, env=env, **opkwargs)


class Sorted:
    def __init__(
        self, op: Callable[..., None], in_place: bool = True, reverse: bool = False
    ) -> None:
        self._op = op
        self._in_place = in_place
        self._reverse = reverse

    @timing
    def __call__(
        self, population: List[Individual], key: str, goal: Goal, env: Environment
    ) -> Any:
        if self._in_place:
            population.sort(
                key=goal.get_cmp_to_key(itemgetter(key)), reverse=self._reverse
            )
            self._op(population, key, goal, env)
        else:
            self._op(
                sorted(
                    population,
                    key=goal.get_cmp_to_key(itemgetter(key)),
                    reverse=self._reverse,
                ),
                key,
                goal,
                env,
            )


def simpleMove(ind: Individual, keyx: str, keyv: str, dt: float) -> None:
    ind[keyx] += dt * ind[keyv]


def l2metrics(x: NDArray, y: NDArray) -> float:
    return float(np.linalg.norm(np.subtract(x, y)))
