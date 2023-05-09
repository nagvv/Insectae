from random import random, randrange
from typing import Tuple
import copy

from .typing import Individual, Evaluable
from .common import evalf
from .targets import Target
from .goals import Goal


class ExpCool:
    def __init__(self, x0: float, q: float) -> None:
        self.x = x0
        self.q = q
        self.gen = 0

    def __call__(self, **kwargs) -> float:
        gen = kwargs['time']
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

    def __call__(self, **kwargs) -> float:
        gen = kwargs['time']
        if gen > self.gen:
            self.gen = gen
            self.x = self.x0 / gen ** self.deg
        return self.x


class RealMutation:
    def __init__(self, delta: Evaluable[float]) -> None:
        self.delta = delta

    def __call__(self, ind: Individual, key: str, **kwargs) -> None:
        delta = evalf(self.delta, inds=[ind], **kwargs)
        dim = len(ind[key])
        for pos in range(dim):
            ind[key][pos] += delta * (1 - 2 * random())


class BinaryMutation:
    def __init__(self, prob: Evaluable[float]) -> None:
        self.prob = prob

    def __call__(self, ind: Individual, key: str, **kwargs) -> None:
        prob = evalf(self.prob, inds=[ind], **kwargs)
        dim = len(ind[key])
        for pos in range(dim):
            if random() < prob:
                ind[key][pos] = 1 - ind[key][pos]


class Tournament:
    def __init__(self, pwin: Evaluable[float]) -> None:
        self.pwin = pwin

    def __call__(self, pair, key: str, twoway: bool, goal: Goal, **kwargs):
        ind1, ind2 = pair
        pwin = evalf(self.pwin, inds=[ind1, ind2], **kwargs)
        A = goal.isBetter(ind1[key], ind2[key])
        B = random() < pwin
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
        **kwargs
    ) -> None:
        ind1, ind2 = pair
        pswap = evalf(self.pswap, inds=[ind1, ind2], **kwargs)
        dim = target.dimension
        for pos in range(dim):
            if random() < pswap:
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
        **kwargs
    ) -> None:
        ind1, ind2 = pair
        dim = target.dimension
        cpos = randrange(1, dim)
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
        **kwargs
    ) -> None:
        ind1, ind2 = pair
        dim = target.dimension
        cpos1 = randrange(1, dim - 1)
        cpos2 = randrange(cpos1, dim)
        for pos in range(cpos1, cpos2):
            if twoway:
                ind1[key][pos], ind2[key][pos] = ind2[key][pos], ind1[key][pos]
            else:
                ind1[key][pos] = ind2[key][pos]


