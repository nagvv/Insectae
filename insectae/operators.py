import copy
from typing import Any, List, Tuple

from .common import evalf
from .goals import Goal
from .targets import Target
from .timer import timing
from .typing import Environment, Evaluable, Individual


class ExpCool:
    def __init__(self, x0: float, q: float) -> None:
        self.x = x0
        self.q = q
        self.gen = 0

    def __call__(self, _: List[Individual], env: Environment) -> float:
        gen = env["time"]
        if gen > self.gen:
            self.gen = gen
            self.x *= self.q
        return self.x  # TODO: добавить возможность работы со списками или кортежами


class HypCool:
    def __init__(self, x0: float, deg: float) -> None:
        self.x0 = x0
        self.x = x0
        self.deg = deg
        self.gen = 0

    def __call__(self, _: List[Individual], env: Environment) -> float:
        gen = env["time"]
        if gen > self.gen:
            self.gen = gen
            self.x = self.x0 / gen**self.deg
        return self.x


class RealMutation:
    def __init__(self, delta: Evaluable[float]) -> None:
        self.delta = delta

    def __call__(self, ind: Individual, key: str, env: Environment) -> None:
        delta = evalf(self.delta, inds=[ind], env=env)
        dim = len(ind[key])
        rng = env["rng"]
        for pos in range(dim):
            ind[key][pos] += delta * (1 - 2 * rng.random())


class BinaryMutation:
    def __init__(self, prob: Evaluable[float]) -> None:
        self.prob = prob

    def __call__(self, ind: Individual, key: str, env: Environment) -> None:
        prob = evalf(self.prob, inds=[ind], env=env)
        dim = len(ind[key])
        rng = env["rng"]
        for pos in range(dim):
            if rng.random() < prob:
                ind[key][pos] = 1 - ind[key][pos]


class Tournament:
    def __init__(self, pwin: Evaluable[float]) -> None:
        self.pwin = pwin

    def __call__(self, pair, key: str, twoway: bool, goal: Goal, env: Environment):
        ind1, ind2 = pair
        pwin = evalf(self.pwin, inds=[ind1, ind2], env=env)
        A = goal.isBetter(ind1[key], ind2[key])
        B = env["rng"].random() < pwin
        if A != B:  # xor
            ind1.update(copy.deepcopy(ind2))
        elif twoway:
            ind2.update(copy.deepcopy(ind1))


class SelectLeft:
    """
    Selects the leftmost `ratio` part of a total population as winners.
    Losers are replaced by winners in a cycle.
    """

    def __init__(self, ratio: Evaluable[float]) -> None:
        self._ratio = ratio

    @timing
    def __call__(
        self, population: List[Individual], key: str, goal: Goal, env: Environment
    ) -> Any:
        count = int(len(population) * evalf(self._ratio, population, env))
        idx = 0
        for loser in population[count:]:
            winner = population[idx]
            loser.update(copy.deepcopy(winner))
            idx = (idx + 1) % count


class UniformCrossover:
    def __init__(self, pswap: Evaluable[float]) -> None:
        self.pswap = pswap

    def __call__(
        self,
        pair: Tuple[Individual, Individual],
        key: str,
        target: Target,
        twoway: bool,
        env: Environment,
    ) -> None:
        ind1, ind2 = pair
        pswap = evalf(self.pswap, inds=[ind1, ind2], env=env)
        dim = target.dimension
        rng = env["rng"]
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
        env: Environment,
    ) -> None:
        ind1, ind2 = pair
        dim = target.dimension
        cpos = env["rng"].integers(1, dim)
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
        env: Environment,
    ) -> None:
        ind1, ind2 = pair
        dim = target.dimension
        cpos1 = env["rng"].integers(1, dim - 1)
        cpos2 = env["rng"].integers(cpos1, dim)
        for pos in range(cpos1, cpos2):
            if twoway:
                ind1[key][pos], ind2[key][pos] = ind2[key][pos], ind1[key][pos]
            else:
                ind1[key][pos] = ind2[key][pos]
