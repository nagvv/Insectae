import copy
from random import choices, random
from typing import Any, Callable, List, Union, TypeVar

import numpy as np

from .patterns import neighbors, pairs
from .typing import Individual, Evaluable

# TODO move some to the operators.py or move these here

_T = TypeVar("_T")


def evalf(param: Evaluable[_T], **opkwargs) -> _T:
    if callable(param):
        return param(**opkwargs)
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


class Mixture:
    def __init__(self, methods: List[Callable[..., None]], probs: List[float]) -> None:
        self.methods = methods + [lambda *args, **kwargs: None]  # append no-op
        self.probs = probs + [1.0 - np.sum(probs)]

    def __call__(self, inds: List[Individual], **opkwargs) -> None:
        method = choices(self.methods, weights=self.probs)[0]
        method(inds, **opkwargs)


class ProbOp:
    def __init__(
        self, method: Callable[..., None], prob: Union[float, Callable[..., float]]
    ) -> None:
        self.method = method
        self.prob = prob

    def __call__(self, inds: List[Individual], **opkwargs) -> None:
        prob = evalf(self.prob, inds=inds, **opkwargs)
        if random() < prob:
            self.method(inds, **opkwargs)


class TimedOp:
    def __init__(self, method: Callable[..., None], dt: int) -> None:
        self.method = method
        self.dt = dt

    def __call__(self, inds: List[Individual], time: int, **opkwargs) -> None:
        if time % self.dt == 0:
            self.method(inds, time=time, **opkwargs)


class Shuffled:
    def __init__(self, op: Callable[..., None]) -> None:
        self.op = op

    def __call__(self, population: List[Individual], **opkwargs) -> None:
        perm = list(range(len(population)))
        np.random.shuffle(perm)
        neighbors(population, self.op, perm, **opkwargs)


def samplex(n: int, m: int, exclude: List[int]) -> List[int]:
    s = list(set(range(n)) - set(exclude))
    return list(np.random.choice(s, m, False))


class Selected:
    def __init__(self, op: Callable[..., None]) -> None:
        self.op = op

    def __call__(self, population: List[Individual], **opkwargs) -> None:
        shadow: List[Individual] = []
        for i in range(len(population)):
            j = samplex(len(population), 1, [i])[0]
            shadow.append(copy.deepcopy(population[j]))
        pairs(population, shadow, self.op, **opkwargs)


def simpleMove(ind: Individual, keyx: str, keyv: str, dt: float) -> None:
    ind[keyx] += dt * ind[keyv]
