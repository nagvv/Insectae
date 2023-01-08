import copy
from random import choices, random
from typing import Any, Callable, Dict, List, Union

import numpy as np

from .patterns import neighbors, pairs


def evalf(param: Any, **opkwargs):
    if callable(param):
        return param(**opkwargs)
    return param


class fillAttribute:
    def __init__(self, op: Any) -> None:
        self.op = op

    def __call__(self, ind: dict, key: str, **opkwargs) -> None:
        if callable(self.op):
            ind[key] = self.op(**opkwargs)
        elif np.isscalar(self.op):
            ind[key] = self.op
        else:
            # FIXME: is there type that is not scalar and doesn't define copy()
            ind[key] = self.op.copy()


def copyAttribute(ind: dict, keyFrom: str, keyTo: str) -> None:
    if np.isscalar(ind[keyFrom]):
        ind[keyTo] = ind[keyFrom]
    else:
        ind[keyTo] = ind[keyFrom].copy()


class mixture:
    def __init__(self, methods: List[Callable[..., None]], probs: List[float]) -> None:
        self.methods = methods + [lambda *args, **kwargs: None]  # append no-op
        self.probs = probs + [1.0 - np.sum(probs)]

    def __call__(self, inds: List[Dict[str, Any]], **opkwargs) -> None:
        method = choices(self.methods, weights=self.probs)[0]
        method(inds, **opkwargs)


class probOp:
    def __init__(
        self, method: Callable[..., None], prob: Union[float, Callable[..., float]]
    ) -> None:
        self.method = method
        self.prob = prob

    def __call__(self, inds: List[Dict[str, Any]], **opkwargs) -> None:
        prob = evalf(self.prob, inds=inds, **opkwargs)
        if random() < prob:
            self.method(inds, **opkwargs)


class timedOp:
    def __init__(self, method: Callable[..., None], dt: int) -> None:
        self.method = method
        self.dt = dt

    def __call__(self, inds: List[Dict[str, Any]], time, **opkwargs) -> None:
        if time % self.dt == 0:
            self.method(inds, **opkwargs)


class shuffled:
    def __init__(self, op: Callable[..., None]) -> None:
        self.op = op

    def __call__(self, population: List[Dict[str, Any]], **opkwargs) -> None:
        perm = list(range(len(population)))
        np.random.shuffle(perm)
        neighbors(population, self.op, perm, **opkwargs)


def samplex(n: int, m: int, exclude: List[int]) -> List[int]:
    s = list(set(range(n)) - set(exclude))
    return list(np.random.choice(s, m, False))


class selected:
    def __init__(self, op: Callable[..., None]) -> None:
        self.op = op

    def __call__(self, population: List[Dict[str, Any]], **opkwargs) -> None:
        shadow: List[Dict[str, Any]] = []
        for i in range(len(population)):
            j = samplex(len(population), 1, [i])[0]
            shadow.append(copy.deepcopy(population[j]))
        pairs(population, shadow, self.op, **opkwargs)


def simpleMove(ind: Dict[str, Any], keyx: str, keyv: str, dt: float) -> None:
    ind[keyx] += dt * ind[keyv]
