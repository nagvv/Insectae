from random import random
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from .alg_base import Algorithm
from .common import Evaluable, copyAttribute, evalf, simpleMove
from .goals import Goal
from .targets import RandomRealVector, RealTarget
from .typing import Environment, Individual


class ParticleSwarmOptimization(Algorithm):
    def __init__(
        self,
        alphabeta: Evaluable[Tuple[float, float]],
        gamma: Evaluable[float],
        delta: Evaluable[float],
        target: RealTarget,
        opLimitVel: Optional[Callable[..., None]] = None,
        **kwargs
    ) -> None:
        self.alphabeta = alphabeta
        self.gamma = gamma
        self.delta = delta
        self.opLimitVel = opLimitVel
        super().__init__(target=target, **kwargs)
        self.target: RealTarget  # hint for type checkers, FIXME is it needed?

    @staticmethod
    def updateVel(ind: Individual, env: Environment) -> None:
        gamma = evalf(env["gamma"], inds=[ind], env=env)
        alpha, beta = evalf(env["alphabeta"], inds=[ind], env=env)
        ind["v"] = (
            gamma * ind["v"]
            + alpha * (ind["p"] - ind["x"])
            + beta * (env["g"] - ind["x"])
        )

    @staticmethod
    def updateBestPosition(ind: Individual, goal: Goal) -> None:
        if goal.isBetter(ind["fNew"], ind["f"]):
            ind["p"] = ind["x"].copy()
            ind["f"] = ind["fNew"]

    def start(self) -> None:
        super().init_attributes("alphabeta gamma g", "&x f *fNew v p")
        self.executor.foreach(
            self.population,
            self.opInit,
            {"target": self.target, "key": "x", "env": self.env},
        )
        self.executor.foreach(
            self.population, copyAttribute, {"keyFrom": "x", "keyTo": "p"}
        )
        self.executor.evaluate(self.population, keyx="x", keyf="f", env=self.env)
        self.executor.foreach(
            self.population, copyAttribute, {"keyFrom": "f", "keyTo": "fNew"}
        )
        delta = evalf(self.delta, inds=self.population, env=self.env)
        vel = delta * (self.target.bounds[1] - self.target.bounds[0])
        self.executor.foreach(
            self.population,
            RandomRealVector((-vel, vel)),
            {"target": self.target, "key": "v"},
        )

    def runGeneration(self) -> None:
        def extract(x: Individual) -> Tuple[Any, Any]:
            return (x["p"], x["f"])

        def op(x, y):
            return x if self.goal.isBetter(x[1], y[1]) else y

        def post(x):
            return x[0]

        timer = self.env.get("timer")
        # TODO add additkwargs, for decorators purpose
        self.env["g"] = self.executor.reducePop(
            self.population, extract, op, post, timingLabel="reduce", timer=timer
        )
        self.executor.foreach(
            self.population,
            self.updateVel,
            {"env": self.env},
            timingLabel="updatevel",
            timer=timer,
        )
        if self.opLimitVel is not None:
            self.executor.foreach(
                self.population,
                self.opLimitVel,
                {
                    "key": "v",
                    "env": self.env,
                },
                timingLabel="limitvel",
                timer=timer,
            )
        self.executor.foreach(
            self.population,
            simpleMove,
            {
                "keyx": "x",
                "keyv": "v",
                "dt": 1.0,
            },
            timingLabel="move",
            timer=timer,
        )
        self.executor.evaluate(
            self.population,
            keyx="x",
            keyf="fNew",
            timingLabel="evaluate",
            timer=timer,
            env=self.env,
        )
        self.executor.foreach(
            self.population,
            self.updateBestPosition,
            {"goal": self.env["goal"]},
            timingLabel="updatebest",
            timer=timer,
        )


class RandomAlphaBeta:
    def __init__(self, alpha: float, beta: float) -> None:
        self.alpha = alpha
        self.beta = beta

    def __call__(self, inds: List[Individual], env: Environment) -> Tuple[float, float]:
        a = random() * self.alpha
        b = random() * self.beta
        return a, b


class LinkedAlphaBeta:
    def __init__(self, total: float) -> None:
        self.total = total

    def __call__(self, inds: List[Individual], env: Environment) -> Tuple[float, float]:
        alpha = random() * self.total
        beta = self.total - alpha
        return alpha, beta


class MaxAmplitude:
    def __init__(self, amax: Evaluable[float]) -> None:
        self.amax = amax

    def __call__(self, ind: Individual, key: str, env: Environment) -> None:
        amax = evalf(self.amax, inds=[ind], env=env)
        a = np.linalg.norm(ind[key])
        if a > amax:
            ind[key] *= amax / a


class FixedAmplitude:
    def __init__(self, ampl: Evaluable[float]) -> None:
        self.ampl = ampl

    def __call__(self, ind: Individual, key: str, env: Environment):
        ampl = evalf(self.ampl, inds=[ind], env=env)
        ind[key] *= ampl / np.linalg.norm(ind[key])
