from math import exp
from operator import add
from sys import float_info
from typing import List
from functools import partial

import numpy as np
from numpy.typing import NDArray

from .alg_base import Algorithm
from .common import evalf, l2metrics
from .targets import RandomRealVector, RealTarget
from .typing import Environment, Evaluable, Individual


class GravitationalSearchAlgorithm(Algorithm):
    def __init__(
        self,
        g_init: float,
        g_decay_stop_it: int,
        alpha: float,
        delta: Evaluable[float],
        **kwargs
    ) -> None:
        self.g_init = g_init
        self.g_decay_stop_it = g_decay_stop_it
        self.alpha = alpha
        self.delta = delta
        super().__init__(**kwargs)
        self.target: RealTarget

    def start(self) -> None:
        super().init_attributes("best worst G sum_m", "&x *f v m M F")
        self.env["G"] = self.g_init
        self.executor.foreach(
            population=self.population,
            op=self.opInit,
            fnkwargs={"target": self.target, "key": "x", "env": self.env},
        )
        self.executor.evaluate(
            population=self.population, keyx="x", keyf="f", env=self.env
        )
        delta = evalf(self.delta, self.env["time"])
        vel = delta * (self.target.bounds[1] - self.target.bounds[0])
        self.executor.foreach(
            self.population,
            RandomRealVector((-vel, vel)),
            {"target": self.target, "key": "v", "env": self.env},
        )

    def update_G(self):
        return self.g_init * exp(-self.alpha * self.env["time"] / self.g_decay_stop_it)

    @staticmethod
    def update_m(x: Individual, best: float, worst: float):
        x["m"] = (x["f"] - worst) / (best - worst + float_info.epsilon)

    @staticmethod
    def update_M(x: Individual, sum_m: float):
        x["M"] = x["m"] / sum_m

    @staticmethod
    def compute_force(dist: float, inds: List[Individual], env: Environment):
        if inds[0] is inds[1]:
            return np.zeros(inds[0]["x"].shape)
        mul = env["G"] * inds[0]["M"] * inds[1]["M"]
        diff = inds[1]["x"] - inds[0]["x"]
        return mul * diff / (dist + float_info.epsilon)

    @staticmethod
    def reduce_force(forces: NDArray, rng: np.random.Generator):
        rnd_shape = (forces.shape[0], *(1,) * (forces.ndim - 1))
        return np.sum(forces * rng.uniform(size=rnd_shape), axis=0)

    @staticmethod
    def move(x: Individual, rng: np.random.Generator):
        vel = rng.random() * x["v"] + x["F"] / (x["M"] + float_info.epsilon)
        x["x"] += vel
        x["v"] = vel

    @staticmethod
    def _extract_ff(x: Individual):
        return (x["f"], x["f"])

    @staticmethod
    def _extract_m(x: Individual):
        return x["m"]

    @staticmethod
    def _reduce(f_1, f_2, goal):
        return (
            f_1[0] if goal.isBetter(f_1[0], f_2[0]) else f_2[0],
            f_2[1] if goal.isBetter(f_1[1], f_2[1]) else f_1[1],
        )

    def runGeneration(self) -> None:
        timer = self.env.get("timer")
        self.env["best"], self.env["worst"] = self.executor.reducePop(
            population=self.population,
            extract=self._extract_ff,
            reduce=partial(self._reduce, goal=self.goal),
            timingLabel="reduce(best,worst)",
            timer=timer,
        )
        self.executor.foreach(
            population=self.population,
            op=self.update_m,
            fnkwargs={"best": self.env["best"], "worst": self.env["worst"]},
            timingLabel="foreach(m)",
            timer=timer,
        )
        self.env["sum_m"] = self.executor.reducePop(
            population=self.population,
            extract=self._extract_m,
            reduce=add,
            timingLabel="reduce(sum_m)",
            timer=timer,
        )
        self.executor.foreach(
            population=self.population,
            op=self.update_M,
            fnkwargs={"sum_m": self.env["sum_m"]},
            timingLabel="foreach(M)",
            timer=timer,
        )
        self.env["G"] = self.update_G()
        self.executor.signals(
            population=self.population,
            metrics=l2metrics,
            shape=self.compute_force,
            reduce=partial(self.reduce_force, rng=self.rng),
            keyx="x",
            keys="F",
            env=self.env,
            timingLabel="signals(F)",
            timer=timer,
        )
        self.executor.foreach(
            population=self.population,
            op=self.move,
            fnkwargs={"rng": self.rng},
            timingLabel="foreach(move)",
            timer=timer,
        )
        self.executor.evaluate(
            population=self.population, keyx="x", keyf="f", env=self.env
        )
