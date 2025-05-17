from typing import Callable, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .alg_base import Algorithm
from .common import evalf, l2metrics
from .operators import copyAttribute, FillAttribute, simpleMove
from .goals import Goal
from .targets import Target
from .typing import Evaluable, Individual, Environment
from .patterns import signals


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
        ind: Individual, dim: int, vel: float, rng: np.random.Generator
    ) -> None:
        ind["v"] = randomDirectedVector(dim, vel, rng)

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
        if new_is_better and r < probRotate[0] or not new_is_better and r < probRotate[1]:
            ind["v"] = randomDirectedVector(dim, vel, rng)

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
            }
        )
        self.executor.evaluate(self.population, keyx="x", keyf="f", env=self.env)
        self.executor.foreach(
            self.population,
            self.initVel,
            {
                "dim": self.target.dimension,
                "vel": evalf(self.vel, self.env["time"]),
                "rng": self.rng,
            },
        )
        self.executor.foreach(
            self.population,
            copyAttribute,
            {
                "keyFrom": "f",
                "keyTo": "fTotal",
            }
        )
        self.executor.foreach(
            self.population,
            FillAttribute(0.0),
            {
                "key": "fs",
                "env": self.env,
            }
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
            self.population, keyx="x", keyf="fNew", timingLabel="evaluate", timer=timer, env=self.env
        )
        self.opSignals(
            self.population,
            key="fs",
            env=self.env,
            timingLabel="signals",
            timer=timer
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
                "vel": evalf(self.vel, self.env["time"]),
                "probRotate": evalf(self.probRotate, self.env["time"]),
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
            {"gamma": evalf(self.gamma, self.env["time"])},
            timingLabel="updatef",
            timer=timer,
        )
        self.opSelect(
            self.population,
            key="fTotal",
            env=self.env,
            timingLabel="select",
            timer=timer
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


# TODO: make it consistent with other helper functions
def randomDirectedVector(dim: int, length: float, rng: np.random.Generator) -> NDArray[np.float64]:
    vec = rng.normal(loc=0.0, scale=1.0, size=dim)
    return vec * (length / np.linalg.norm(vec))


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

    def __call__(
        self, population: List[Individual], key: str, env: Environment, **kwargs
    ) -> None:
        env["executor"].signals(
            population=population,
            metrics=self.metrics,
            shape=self.shape,
            reduce=self.reduce,
            keyx="x",  # TODO will be removed? or pass as a pair
            keys=key,
            env=env,
            **kwargs
        )


# Different signal shapes


class ShapeClustering:
    def __init__(self, d: Evaluable[float], goal: Union[Goal, str] = "min") -> None:
        self.d = d
        self.goal = 1 if goal == "min" else -1

    def __call__(self, x: float, inds: List[Individual], env: Environment) -> float:
        d = evalf(self.d, env["time"])
        x2 = (x / d) ** 2
        return self.goal * (2 * np.exp(-x2) - 3 * np.exp(-4 * x2))
