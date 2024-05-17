from typing import Callable, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .alg_base import Algorithm
from .common import copyAttribute, evalf, FillAttribute, simpleMove, l2metrics
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
        ind: Individual, vel: Evaluable[float], target: Target, env: Environment
    ) -> None:
        dim = target.dimension
        vel = evalf(vel, inds=[ind], env=env)
        ind["v"] = randomDirectedVector(dim, vel, env["rng"])

    @staticmethod
    def rotate(
        ind: Individual,
        vel: Evaluable[float],
        probRotate: Evaluable[Tuple[float, float]],
        target: Target,
        goal: Goal,
        env: Environment,
    ) -> None:
        prob = evalf(probRotate, inds=[ind], env=env)
        new_is_better = goal.isBetter(ind["fNew"], ind["f"])
        r = env["rng"].random()
        if new_is_better and r < prob[0] or not new_is_better and r < prob[1]:
            vel = evalf(vel, inds=[ind], env=env)
            dim = target.dimension
            ind["v"] = randomDirectedVector(dim, vel, env["rng"])

    @staticmethod
    def updateF(ind: Individual, gamma: Evaluable[float], env: Environment) -> None:
        gamma = evalf(gamma, inds=[ind], env=env)
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
                "vel": self.vel,
                "target": self.target,
                "env": self.env,
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
            keyx="x",
            keys="fs",
            timingLabel="signals",
            timer=timer,
            env=self.env
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
                "vel": self.vel,
                "probRotate": self.probRotate,
                "goal": self.goal,
                "target": self.target,
                "env": self.env,
            },
            timingLabel="rotate",
            timer=timer,
        )
        self.executor.foreach(
            self.population,
            self.updateF,
            {
                "gamma": self.gamma,
                "env": self.env,
            },
            timingLabel="updatef",
            timer=timer,
        )
        self.opSelect(
            self.population,
            key="fTotal",
            goal=self.goal,
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

    def __call__(self, population: List[Individual], keyx: str, keys: str, **kwargs) -> None:
        # TODO pass executor
        signals(
            population=population,
            metrics=self.metrics,
            shape=self.shape,
            reduce=self.reduce,
            keyx=keyx,
            keys=keys,
            **kwargs
        )


# Different signal shapes


class ShapeClustering:
    def __init__(self, d: Evaluable[float], goal: Union[Goal, str] = "min") -> None:
        self.d = d
        self.goal = 1 if goal == "min" else -1

    def __call__(self, x: float, inds: List[Individual], env: Environment) -> float:
        d = evalf(self.d, inds=inds, env=env)
        x2 = (x / d) ** 2
        return self.goal * (2 * np.exp(-x2) - 3 * np.exp(-4 * x2))
