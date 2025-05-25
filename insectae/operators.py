import copy
from functools import partial
from operator import itemgetter
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np

from .common import evalf, get_args_from_env, samplex, weighted_choice
from .goals import Goal
from .targets import Target
from .timer import timing
from .typing import Environment, Evaluable, Individual


# Whole population operators
# They have fixed call signature: population, key, env, **kwargs
# kwargs are commonly used to pass timer and a label to an executor
class SelectLeft:
    """
    Selects the leftmost `ratio` part of a total population as winners.
    Losers are replaced by winners in a cycle.
    """

    def __init__(self, ratio: Evaluable[float]) -> None:
        self._ratio = ratio

    @timing
    def __call__(
        self, population: List[Individual], key: str, env: Environment, **kwargs
    ) -> None:
        count = int(len(population) * evalf(self._ratio, env["time"], env["rng"]))
        idx = 0
        for loser in population[count:]:
            winner = population[idx]
            loser.update(copy.deepcopy(winner))
            idx = (idx + 1) % count


class Sorted:
    def __init__(
        self, op: Callable[..., None], in_place: bool = True, reverse: bool = False
    ) -> None:
        self.op = op
        self.in_place = in_place
        self.reverse = reverse

    @timing
    def __call__(
        self, population: List[Individual], key: str, env: Environment, **kwargs
    ) -> None:
        goal = env["goal"]
        if self.in_place:
            population.sort(
                key=goal.get_cmp_to_key(itemgetter(key)), reverse=self.reverse
            )
            self.op(population, key=key, env=env, **kwargs)
        else:
            self.op(
                sorted(
                    population,
                    key=goal.get_cmp_to_key(itemgetter(key)),
                    reverse=self.reverse,
                ),
                key=key,
                env=env,
                **kwargs,
            )


class ShuffledNeighbors:
    def __init__(
        self,
        op: Callable[..., None],
        args_from_env_getter: Callable[[Environment], Dict[str, Any]] = None,
    ) -> None:
        self.op = op
        self.args_from_env_getter = args_from_env_getter
        if self.args_from_env_getter is None:
            self.args_from_env_getter = partial(get_args_from_env, self.op)

    def __call__(
        self, population: List[Individual], key: str, env: Environment, **kwargs
    ) -> None:
        rng = env["rng"]
        perm = list(range(len(population)))
        rng.shuffle(perm)
        env["executor"].neighbors(
            population=population,
            op=self.op,
            permutation=perm,
            fnkwargs={"key": key, **self.args_from_env_getter(env)},
            **kwargs,
        )


class TimedOp:
    def __init__(self, op: Callable[..., None], dt: int) -> None:
        self.op = op
        self.dt = dt

    def __call__(
        self, population: List[Individual], key: str, env: Environment, **kwargs
    ) -> None:
        time = env["time"]
        if time % self.dt == 0:
            self.op(population, key=key, env=env, **kwargs)


class ProbOp:
    def __init__(
        self, op: Callable[..., None], prob: Union[float, Callable[..., float]]
    ) -> None:
        self.op = op
        self.prob = prob

    def __call__(
        self, population: List[Individual], key: str, env: Environment, **kwargs
    ) -> None:
        prob = evalf(self.prob, env["time"], env["rng"])
        if env["rng"].random() < prob:
            self.op(population, key=key, env=env, **kwargs)


class Selected:
    def __init__(
        self,
        op: Callable[..., None],
        args_from_env_getter: Callable[[Environment], Dict[str, Any]] = None,
    ) -> None:
        self.op = op
        self.args_from_env_getter = args_from_env_getter
        if self.args_from_env_getter is None:
            self.args_from_env_getter = partial(get_args_from_env, self.op)

    def __call__(
        self, population: List[Individual], key: str, env: Environment, **kwargs
    ) -> None:
        shadow: List[Individual] = []
        rng = env["rng"]
        for i in range(len(population)):
            j = samplex(len(population), 1, [i], rng)[0]
            shadow.append(copy.deepcopy(population[j]))
        env["executor"].pairs(
            population,
            shadow,
            self.op,
            fnkwargs={
                "key": key,
                **self.args_from_env_getter(env),
            },
            **kwargs,
        )


class Mixture:
    def __init__(self, ops: List[Callable[..., None]], probs: List[float]) -> None:
        if len(ops) != len(probs):
            raise ValueError("The lengths of `ops` and `probs` must be the same.")
        if any(w < 0 for w in probs):
            raise ValueError("`probs` must be non-negative.")
        if sum(probs) == 0:
            raise ValueError("The sum of probs must be greater than zero.")

        self.ops = ops + [self._noop]  # append no-op
        self.probs = probs + [1.0 - np.sum(probs)]

    @staticmethod
    def _noop(*args, **kwargs):
        pass

    def __call__(
        self, population: List[Individual], key: str, env: Environment, **kwargs
    ) -> None:
        op = weighted_choice(self.ops, self.probs, env["rng"])
        op(population, key=key, env=env, **kwargs)


# Per-individual/pair operators
# Their signature is not strictly fixed
class RealMutation:
    def __init__(self, delta: Evaluable[float]) -> None:
        self.delta = delta

    def __call__(
        self, ind: Individual, key: str, time: int, rng: np.random.Generator
    ) -> None:
        delta = evalf(self.delta, time, rng)
        dim = len(ind[key])
        rng = rng
        for pos in range(dim):
            ind[key][pos] += delta * (1 - 2 * rng.random())


class BinaryMutation:
    def __init__(self, prob: Evaluable[float]) -> None:
        self.prob = prob

    def __call__(
        self, ind: Individual, key: str, time: int, rng: np.random.Generator
    ) -> None:
        prob = evalf(self.prob, time, rng)
        dim = len(ind[key])
        rng = rng
        for pos in range(dim):
            if rng.random() < prob:
                ind[key][pos] = 1 - ind[key][pos]


class Tournament:
    def __init__(self, pwin: Evaluable[float]) -> None:
        self.pwin = pwin

    def __call__(
        self,
        pair,
        key: str,
        twoway: bool,
        goal: Goal,
        time: int,
        rng: np.random.Generator,
    ):
        ind1, ind2 = pair
        pwin = evalf(self.pwin, time, rng)
        A = goal.isBetter(ind1[key], ind2[key])
        B = rng.random() < pwin
        if A != B:  # xor
            ind1.update(copy.deepcopy(ind2))
        elif twoway:
            ind2.update(copy.deepcopy(ind1))


class UniformCrossover:
    def __init__(self, pswap: Evaluable[float]) -> None:
        self.pswap = pswap

    def __call__(
        self,
        pair: Tuple[Individual, Individual],
        key: str,
        target: Target,
        twoway: bool,
        time: int,
        rng: np.random.Generator,
    ) -> None:
        ind1, ind2 = pair
        pswap = evalf(self.pswap, time, rng)
        dim = target.dimension
        rng = rng
        for pos in range(dim):
            if rng.random() < pswap:
                if twoway:
                    ind1[key][pos], ind2[key][pos] = ind2[key][pos], ind1[key][pos]
                else:
                    ind1[key][pos] = ind2[key][pos]


class SinglePointCrossover:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        pair: Tuple[Individual, Individual],
        key: str,
        target: Target,
        twoway: bool,
        rng: np.random.Generator,
    ) -> None:
        ind1, ind2 = pair
        dim = target.dimension
        cpos = rng.integers(1, dim)
        for pos in range(cpos, dim):
            if twoway:
                ind1[key][pos], ind2[key][pos] = ind2[key][pos], ind1[key][pos]
            else:
                ind1[key][pos] = ind2[key][pos]


class DoublePointCrossover:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        pair: Tuple[Individual, Individual],
        key: str,
        target: Target,
        twoway: bool,
        rng: np.random.Generator,
    ) -> None:
        ind1, ind2 = pair
        dim = target.dimension
        cpos1 = rng.integers(1, dim - 1)
        cpos2 = rng.integers(cpos1, dim)
        for pos in range(cpos1, cpos2):
            if twoway:
                ind1[key][pos], ind2[key][pos] = ind2[key][pos], ind1[key][pos]
            else:
                ind1[key][pos] = ind2[key][pos]


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


def simpleMove(ind: Individual, keyx: str, keyv: str, dt: float) -> None:
    ind[keyx] += dt * ind[keyv]
