from typing import Any
import numpy as np
from random import choices, random
from .patterns_dirty import neighbors, pairs
import copy


def evalf(param: Any, **xt):
    if callable(param):
        return param(**xt)
    return param


class fillAttribute:
    def __init__(self, op):
        self.op = op

    def __call__(self, ind, **xt):
        key = xt["key"]
        if callable(self.op):
            ind[key] = self.op(**xt)
        elif np.isscalar(self.op):
            ind[key] = self.op
        else:
            ind[key] = self.op.copy()


def copyAttribute(ind, **xt):
    keyFrom = xt["keyFrom"]
    keyTo = xt["keyTo"]
    if np.isscalar(ind[keyFrom]):
        ind[keyTo] = ind[keyFrom]
    else:
        ind[keyTo] = ind[keyFrom].copy()


class mixture:
    def __init__(self, methods, probs):
        self.methods = methods + [None]
        self.probs = probs + [1 - np.sum(probs)]

    def __call__(self, inds, **xt):
        m = choices(self.methods, weights=self.probs)[0]
        if m:
            m(inds, **xt)


class probOp:
    def __init__(self, method, prob):
        self.method = method
        self.prob = prob

    def __call__(self, inds, **xt):
        prob = evalf(self.prob, inds=inds, **xt)
        if random() < prob:
            self.method(inds, **xt)


class timedOp:
    def __init__(self, method, dt):
        self.method = method
        self.dt = dt

    def __call__(self, inds, **xt):
        t = xt["time"]
        if t % self.dt == 0:
            self.method(inds, **xt)


class shuffled:
    def __init__(self, op):
        self.op = op

    def __call__(self, population, **xt):
        P = list(range(len(population)))
        np.random.shuffle(P)
        neighbors(population, self.op, P, **xt)


class selected:
    def __init__(self, op):
        self.op = op

    def __call__(self, population, **xt):
        shadow = []
        for i in range(len(population)):
            j = samplex(len(population), 1, [i])[0]
            shadow.append(copy.deepcopy(population[j]))
        pairs(population, shadow, self.op, **xt)


def samplex(n, m, x):
    s = list(set(range(n)) - set(x))
    return list(np.random.choice(s, m, False))


def simpleMove(ind, **xt):
    keyx = xt["keyx"]
    keyv = xt["keyv"]
    dt = xt["dt"]
    ind[keyx] += dt * ind[keyv]
