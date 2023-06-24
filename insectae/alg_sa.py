from random import random
from typing import Callable

from numpy import exp

from .alg_base import Algorithm
from .common import copyAttribute, evalf
from .goals import Goal
from .typing import Environment, Evaluable, Individual


class SimulatedAnnealing(Algorithm):
    def __init__(
        self,
        theta: Evaluable[float],
        opMove: Callable[..., None],
        **kwargs,
    ) -> None:
        self.theta = theta
        self.opMove = opMove
        super().__init__(**kwargs)

    def start(self) -> None:
        super().init_attributes("theta", "&x xNew *f fNew")
        self.executor.foreach(
            self.population,
            self.opInit,
            {"target": self.target, "key": "x", "env": self.env},
        )
        self.executor.evaluate(self.population, keyx="x", keyf="f", env=self.env)

    def runGeneration(self) -> None:
        timer = self.env.get("timer")
        self.executor.foreach(
            self.population,
            copyAttribute,
            {"keyFrom": "x", "keyTo": "xNew"},
            timingLabel="copy",
            timer=timer,
        )
        self.executor.foreach(
            self.population,
            self.opMove,
            {"key": "xNew", "env": self.env},
            timingLabel="move",
            timer=timer,
        )
        self.executor.evaluate(
            self.population,
            keyx="xNew",
            keyf="fNew",
            timingLabel="evaluate",
            timer=timer,
            env=self.env,
        )
        self.executor.foreach(
            self.population,
            self.accept,
            {
                "theta": self.theta,
                "goal": self.goal,
                "env": self.env,
            },
            timingLabel="accept",
            timer=timer,
        )

    @staticmethod
    def accept(
        ind: Individual, theta: Evaluable[float], goal: Goal, env: Environment
    ) -> None:
        theta = evalf(theta, inds=[ind], env=env)
        df = abs(ind["fNew"] - ind["f"])
        if goal.isBetter(ind["fNew"], ind["f"]) or random() < exp(-df / theta):
            ind["f"] = ind["fNew"]
            ind["x"] = ind["xNew"].copy()
