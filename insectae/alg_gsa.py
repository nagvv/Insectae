from math import exp
from operator import add
from sys import float_info
from typing import List
from functools import partial

import numpy as np

from .alg_base import Algorithm
from .common import evalf, l2metrics
from .targets import RandomRealVector, RealTarget
from .typing import Evaluable, Individual


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
            population=self.population, keyx="x", keyf="f", target=self.target
        )
        delta = evalf(self.delta, self.env["time"], self.rng)
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
    def compute_force(pair, metrics, G):
        ind1, ind2 = pair
        dist = metrics(ind1["x"], ind2["x"])
        mul = G * ind1["M"] * ind2["M"]
        # this is not a complete force; the lacking directional vector will be
        # added in the next step, i.e. 'reduce_force'
        return mul / (dist + float_info.epsilon)

    @staticmethod
    def _reduce_force(ind, paired_forces: List, rng: np.random.Generator):
        forces = (force * (ind2["x"] - ind["x"]) for force, ind2 in paired_forces)
        weights = rng.uniform(size=(len(paired_forces),))
        ind["F"] = np.sum([force * w for force, w in zip(forces, weights)], axis=0)

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
        self.executor.allNeighbors(
            self.population,
            op=self.compute_force,
            op_fnkwargs={"metrics": l2metrics, "G": self.env["G"]},
            post=self._reduce_force,
            post_fnkwargs={"rng": self.rng},
            timingLabel="forces(F)",
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
            population=self.population, keyx="x", keyf="f", target=self.target
        )
