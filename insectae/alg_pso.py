from functools import partial
from typing import Any, Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .alg_base import Algorithm
from .common import Evaluable, evalf
from .goals import Goal
from .operators import copyAttribute, simpleMove
from .targets import RandomRealVector, RealTarget
from .typing import Individual


class ParticleSwarmOptimization(Algorithm):
    def __init__(
        self,
        alphabeta: Evaluable[Tuple[float, float]],
        gamma: Evaluable[float],
        delta: Evaluable[float],
        target: RealTarget,
        opLimitVel: Optional[Callable[..., None]] = None,
        **kwargs,
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
        g: NDArray,
        gamma: float,
        alphabeta: Tuple[float, float],
    ) -> None:
        alpha, beta = alphabeta
        ind["v"] = (
            gamma * ind["v"] + alpha * (ind["p"] - ind["x"]) + beta * (g - ind["x"])
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
        self.executor.evaluate(self.population, keyx="x", keyf="f", target=self.target)
        self.executor.foreach(
            self.population, copyAttribute, {"keyFrom": "f", "keyTo": "fNew"}
        )
        delta = evalf(self.delta, self.env["time"], self.rng)
        vel = delta * (self.target.bounds[1] - self.target.bounds[0])
        self.executor.foreach(
            self.population,
            RandomRealVector((-vel, vel)),
            {"target": self.target, "key": "v", "env": self.env},
        )

    @staticmethod
    def _extract(x: Individual) -> Tuple[Any, Any]:
        return (x["p"], x["f"])

    @staticmethod
    def _reduce(x, y, goal):
        # [0] - solution value, [1] - fitness value
        return x if goal.isBetter(x[1], y[1]) else y

    def runGeneration(self) -> None:
        timer = self.env.get("timer")
        self.env["g"] = self.executor.reducePop(
            self.population,
            self._extract,
            partial(self._reduce, goal=self.goal),
            timingLabel="reduce",
            timer=timer,
        )[0]
        self.executor.foreach(
            self.population,
            self.updateVel,
            {
                "g": self.env["g"],
                "gamma": evalf(self.gamma, self.env["time"], self.rng),
                "alphabeta": evalf(self.alphabeta, self.env["time"], self.rng),
            },
            timingLabel="updatevel",
            timer=timer,
        )
        if self.opLimitVel is not None:
            self.executor.foreach(
                self.population,
                self.opLimitVel,
                {
                    "key": "v",
                    "time": self.env["time"],
                    "rng": self.rng,
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
            target=self.target,
        )
        self.executor.foreach(
            self.population,
            self.updateBestPosition,
            {"goal": self.goal},
            timingLabel="updatebest",
            timer=timer,
        )


class RandomAlphaBeta:
    def __init__(self, alpha: float, beta: float) -> None:
        self.alpha = alpha
        self.beta = beta

    def __call__(self, time: int, rng: np.random.Generator) -> Tuple[float, float]:
        a = rng.random() * self.alpha
        b = rng.random() * self.beta
        return a, b


class LinkedAlphaBeta:
    def __init__(self, total: float) -> None:
        self.total = total

    def __call__(self, time: int, rng: np.random.Generator) -> Tuple[float, float]:
        alpha = rng.random() * self.total
        beta = self.total - alpha
        return alpha, beta


class MaxAmplitude:
    def __init__(self, amax: Evaluable[float]) -> None:
        self.amax = amax

    def __call__(
        self, ind: Individual, key: str, time: int, rng: np.random.Generator
    ) -> None:
        amax = evalf(self.amax, time, rng)
        a = np.linalg.norm(ind[key])
        if a > amax:
            ind[key] *= amax / a


class FixedAmplitude:
    def __init__(self, ampl: Evaluable[float]) -> None:
        self.ampl = ampl

    def __call__(
        self, ind: Individual, key: str, time: int, rng: np.random.Generator
    ) -> None:
        ampl = evalf(self.ampl, time, rng)
        ind[key] *= ampl / np.linalg.norm(ind[key])
