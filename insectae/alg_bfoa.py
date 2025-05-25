from typing import Callable, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .alg_base import Algorithm
from .common import evalf, l2metrics
from .goals import Goal
from .operators import FillAttribute, copyAttribute, simpleMove
from .typing import Environment, Evaluable, Individual


def _randomDirectedVector(
    dim: int, length: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    vec = rng.normal(loc=0.0, scale=1.0, size=dim)
    return vec * (length / np.linalg.norm(vec))


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
        **kwargs,
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
        ind: Individual, dim: int, vel: float, rng: np.random.Generator
    ) -> None:
        ind["v"] = _randomDirectedVector(dim, vel, rng)

    @staticmethod
    def rotate(
        ind: Individual,
        vel: float,
        probRotate: Tuple[float, float],
        dim: int,
        goal: Goal,
        rng: np.random.Generator,
    ) -> None:
        new_is_better = goal.isBetter(ind["fNew"], ind["f"])
        r = rng.random()
        if (
            new_is_better
            and r < probRotate[0]
            or not new_is_better
            and r < probRotate[1]
        ):
            ind["v"] = _randomDirectedVector(dim, vel, rng)

    @staticmethod
    def updateF(ind: Individual, gamma: float) -> None:
        ind["f"] = ind["fNew"]
        ind["fTotal"] = (gamma * ind["fTotal"] + ind["fNew"]) / (gamma + 1)

    def start(self) -> None:
        super().init_attributes("vel gamma probRotate", "&x *f fNew fs fTotal v")
        self.executor.foreach(
            self.population,
            self.opInit,
            {
                "target": self.target,
                "key": "x",
                "env": self.env,
            },
        )
        self.executor.evaluate(self.population, keyx="x", keyf="f", target=self.target)
        self.executor.foreach(
            self.population,
            self.initVel,
            {
                "dim": self.target.dimension,
                "vel": evalf(self.vel, self.env["time"], self.rng),
                "rng": self.rng,
            },
        )
        self.executor.foreach(
            self.population,
            copyAttribute,
            {
                "keyFrom": "f",
                "keyTo": "fTotal",
            },
        )
        self.executor.foreach(
            self.population,
            FillAttribute(0.0),
            {
                "key": "fs",
                "env": self.env,
            },
        )

    def runGeneration(self) -> None:
        timer = self.env.get("timer")
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
        self.opSignals(
            self.population, key="fs", env=self.env, timingLabel="signals", timer=timer
        )
        self.executor.foreach(
            self.population,
            simpleMove,
            {
                "keyx": "fNew",
                "keyv": "fs",
                "dt": self.mu,
            },
            timingLabel="newf",
            timer=timer,
        )
        self.executor.foreach(
            self.population,
            self.rotate,
            {
                "vel": evalf(self.vel, self.env["time"], self.rng),
                "probRotate": evalf(self.probRotate, self.env["time"], self.rng),
                "goal": self.goal,
                "dim": self.target.dimension,
                "rng": self.rng,
            },
            timingLabel="rotate",
            timer=timer,
        )
        self.executor.foreach(
            self.population,
            self.updateF,
            {"gamma": evalf(self.gamma, self.env["time"], self.rng)},
            timingLabel="updatef",
            timer=timer,
        )
        self.opSelect(
            self.population,
            key="fTotal",
            env=self.env,
            timingLabel="select",
            timer=timer,
        )
        self.executor.foreach(
            self.population,
            self.opDisperse,
            {
                "key": "x",
                "env": self.env,
                "target": self.target,
            },
            timingLabel="disperse",
            timer=timer,
        )


class NoSignals:
    def __call__(
        self, population: List[Individual], key: str, env: Environment, **kwargs
    ) -> None:
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

    @staticmethod
    def _op(pair, metrics, shape, time: int, rng: np.random.Generator):
        return shape(metrics(pair[0], pair[1]), time=time, rng=rng)

    @staticmethod
    def _reduce(ind, paired_values, reduce, key):
        ind[key] = reduce(v for v, _ in paired_values)

    def __call__(
        self, population: List[Individual], key: str, env: Environment, **kwargs
    ) -> None:
        env["executor"].allNeighbors(
            population=population,
            op=self._op,
            op_fnkwargs={
                "metrics": self.metrics,
                "shape": self.shape,
                "time": env["time"],
                "rng": env["rng"],
            },
            op_getter="x",
            post=self._reduce,
            post_fnkwargs={"key": key, "reduce": self.reduce},
            **kwargs,
        )


class ShapeClustering:
    def __init__(self, d: Evaluable[float], goal: Union[Goal, str] = "min") -> None:
        self.d = d
        self.goal = 1 if goal == "min" else -1

    def __call__(self, x: float, time: int, rng: np.random.Generator) -> float:
        d = evalf(self.d, time, rng)
        x2 = (x / d) ** 2
        return self.goal * (2 * np.exp(-x2) - 3 * np.exp(-4 * x2))
