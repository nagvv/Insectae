from random import random
from typing import Callable, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .alg_base import Algorithm
from .common import copyAttribute, evalf, FillAttribute, simpleMove
from .goals import Goal
from .patterns import evaluate, foreach, signals
from .targets import Target
from .typing import Evaluable, Individual


class BacterialForagingAlgorithm(Algorithm):
    def __init__(
        self,
        vel: Evaluable[float],
        gamma: Evaluable[float],
        probRotate: Evaluable[float],
        mu: float,
        opSelect: Callable[..., None],
        opDisperse: Callable[..., None],
        opSignals: Callable[..., None],
        **kwargs
    ) -> None:
        self.vel = vel
        self.gamma = gamma
        self.probRotate = probRotate
        self.mu = mu
        self.opSelect = opSelect
        self.opDisperse = opDisperse
        self.opSignals = opSignals
        super().__init__(**kwargs)

    @staticmethod
    def initVel(
        ind: Individual, vel: Evaluable[float], target: Target, **kwargs
    ) -> None:
        dim = target.dimension
        vel = evalf(vel, inds=[ind], **kwargs)
        ind["v"] = randomDirectedVector(dim, vel)

    @staticmethod
    def rotate(
        ind: Individual,
        vel: Evaluable[float],
        probRotate: Evaluable[Tuple[float, float]],
        target: Target,
        goal: Goal,
        **kwargs
    ) -> None:
        prob = evalf(probRotate, inds=[ind], **kwargs)
        new_is_better = goal.isBetter(ind["fNew"], ind["f"])
        r = random()
        if new_is_better and r < prob[0] or not new_is_better and r < prob[1]:
            vel = evalf(vel, inds=[ind], **kwargs)
            dim = target.dimension
            ind["v"] = randomDirectedVector(dim, vel)

    @staticmethod
    def updateF(ind: Individual, gamma: Evaluable[float], **kwargs) -> None:
        gamma = evalf(gamma, inds=[ind], **kwargs)
        ind["f"] = ind["fNew"]
        ind["fTotal"] = (gamma * ind["fTotal"] + ind["fNew"]) / (gamma + 1)

    def start(self) -> None:
        super().init_attributes("vel gamma probRotate", "&x *f fNew fs fTotal v")
        foreach(self.population, self.opInit, key="x", **self.env)
        evaluate(self.population, keyx="x", keyf="f", env=self.env)
        foreach(self.population, self.initVel, **self.env)
        foreach(self.population, copyAttribute, keyFrom="f", keyTo="fTotal")
        foreach(self.population, FillAttribute(0.0), key="fs", **self.env)

    def runGeneration(self) -> None:
        foreach(
            self.population,
            simpleMove,
            keyx="x",
            keyv="v",
            dt=1.0,
            timingLabel="move"
        )
        evaluate(
            self.population, keyx="x", keyf="fNew", timingLabel="evaluate", env=self.env
        )
        self.opSignals(
            self.population,
            keyx="x",
            keys="fs",
            timingLabel="signals",
            **self.env
        )
        foreach(
            self.population,
            simpleMove,
            keyx="fNew",
            keyv="fs",
            dt=self.mu,
            timingLabel="newf"
        )
        foreach(self.population, self.rotate, timingLabel="rotate", **self.env)
        foreach(self.population, self.updateF, timingLabel="updatef", **self.env)
        self.opSelect(self.population, key="fTotal", timingLabel="select", **self.env)
        foreach(
            self.population,
            self.opDisperse,
            key="x",
            timingLabel="disperse",
            **self.env
        )


def randomDirectedVector(dim: int, length: float) -> NDArray[np.float64]:
    vec = np.random.normal(0.0, 1.0, size=dim)
    return vec * (length / np.linalg.norm(vec))


def l2metrics(x: NDArray, y: NDArray) -> float:
    return float(np.linalg.norm(np.subtract(x, y)))


class NoSignals:
    def __call__(self, population: List[Individual], **kwargs) -> None:
        pass


class CalcSignals:
    def __init__(
        self,
        shape: Callable[..., float],
        reduce: Union[Callable[..., float], str] = np.sum,
        metrics: Callable[..., float] = l2metrics,
    ) -> None:
        self.shape = shape
        if isinstance(reduce, str):
            reduce_ops = {"sum": np.sum, "min": np.min, "max": np.max, "mean": np.mean}
            if reduce in reduce_ops:
                self.reduce = reduce_ops[reduce]
            else:
                raise ValueError("unknown op name is provided")
        else:
            self.reduce = reduce
        self.metrics = metrics

    def __call__(self, population: List[Individual], keyx: str, keys: str, **kwargs) -> None:
        signals(
            population=population,
            metrics=self.metrics,
            shape=self.shape,
            reduce=self.reduce,
            keyx=keyx,
            keys=keys,
            env=kwargs,
        )


# Different signal shapes


class ShapeClustering:
    def __init__(self, d: Evaluable[float], goal: Union[Goal, str] = "min") -> None:
        self.d = d
        self.goal = 1 if goal == "min" else -1

    def __call__(self, x: float, **kwargs) -> float:
        d = evalf(self.d, **kwargs)
        x2 = (x / d) ** 2
        return self.goal * (2 * np.exp(-x2) - 3 * np.exp(-4 * x2))
