from random import random
from typing import Callable, Tuple

import numpy as np

from .alg_base import Algorithm
from .common import Evaluable, copyAttribute, evalf, simpleMove
from .goals import Goal
from .patterns import evaluate, foreach, reducePop
from .targets import RandomRealVector, RealTarget
from .typing import Individual


class ParticleSwarmOptimization(Algorithm):
    def __init__(
        self,
        alphabeta: Evaluable[Tuple[float, float]],
        gamma: Evaluable[float],
        delta: Evaluable[float],
        target: RealTarget,
        opLimitVel: Callable[..., None] = lambda ind, **kwargs: None,
        **kwargs
    ) -> None:
        self.alphabeta = alphabeta
        self.gamma = gamma
        self.delta = delta
        self.opLimitVel = opLimitVel
        super().__init__(target=target, **kwargs)
        self.target: RealTarget  # hint for type checkers, FIXME is it needed?

    @staticmethod
    def updateVel(
        ind: Individual,
        alphabeta: Evaluable[Tuple[float, float]],
        gamma: Evaluable[float],
        g: Evaluable[float],
        **kwargs
    ) -> None:
        gamma = evalf(gamma, inds=[ind], **kwargs)
        alpha, beta = evalf(alphabeta, inds=[ind], **kwargs)
        ind["v"] = (
            gamma * ind["v"] + alpha * (ind["p"] - ind["x"]) + beta * (g - ind["x"])
        )

    @staticmethod
    def updateBestPosition(ind: Individual, goal: Goal) -> None:
        if goal.isBetter(ind["fNew"], ind["f"]):
            ind["p"] = ind["x"].copy()
            ind["f"] = ind["fNew"]

    def start(self) -> None:
        super().start("alphabeta gamma g", "&x f *fNew v p")
        foreach(self.population, self.opInit, key="x", **self.env)
        foreach(self.population, copyAttribute, keyFrom="x", keyTo="p")
        evaluate(self.population, keyx="x", keyf="f", env=self.env)
        foreach(self.population, copyAttribute, keyFrom="f", keyTo="fNew")
        delta = evalf(self.delta, inds=self.population, **self.env)
        vel = delta * (self.target.bounds[1] - self.target.bounds[0])
        foreach(
            self.population,
            RandomRealVector((-vel, vel)),
            target=self.target,
            key="v"
        )

    def runGeneration(self) -> None:
        extract = lambda x: (x["p"], x["f"])
        op = lambda x, y: x if self.goal.isBetter(x[1], y[1]) else y
        post = lambda x: x[0]
        self.env["g"] = reducePop(
            self.population, extract, op, post, timingLabel="reduce"
        )
        foreach(self.population, self.updateVel, timingLabel="updatevel", **self.env)
        foreach(
            self.population,
            self.opLimitVel,
            key="v",
            timingLabel="limitvel",
            **self.env
        )
        foreach(
            self.population, simpleMove, keyx="x", keyv="v", dt=1.0, timingLabel="move"
        )
        evaluate(
            self.population, keyx="x", keyf="fNew", timingLabel="evaluate", env=self.env
        )
        foreach(
            self.population,
            self.updateBestPosition,
            timingLabel="updatebest",
            goal=self.env["goal"],
        )


class RandomAlphaBeta:
    def __init__(self, alpha: float, beta: float) -> None:
        self.alpha = alpha
        self.beta = beta

    def __call__(self, **kwargs) -> Tuple[float, float]:
        a = random() * self.alpha
        b = random() * self.beta
        return a, b


class LinkedAlphaBeta:
    def __init__(self, total: float) -> None:
        self.total = total

    def __call__(self, **kwargs) -> Tuple[float, float]:
        alpha = random() * self.total
        beta = self.total - alpha
        return alpha, beta


class MaxAmplitude:
    def __init__(self, amax: Evaluable[float]) -> None:
        self.amax = amax

    def __call__(self, ind: Individual, key: str, **kwargs) -> None:
        amax = evalf(self.amax, inds=[ind], **kwargs)
        a = np.linalg.norm(ind[key])
        if a > amax:
            ind[key] *= amax / a


class FixedAmplitude:
    def __init__(self, ampl: Evaluable[float]) -> None:
        self.ampl = ampl

    def __call__(self, ind: Individual, key: str, **kwargs):
        ampl = evalf(self.ampl, inds=[ind], **kwargs)
        ind[key] *= ampl / np.linalg.norm(ind[key])
