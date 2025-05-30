from typing import Callable

from numpy import exp, random

from .alg_base import Algorithm
from .common import evalf
from .goals import Goal
from .operators import copyAttribute
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
        self.executor.evaluate(self.population, keyx="x", keyf="f", target=self.target)

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
            {"key": "xNew", "time": self.env["time"], "rng": self.rng},
            timingLabel="move",
            timer=timer,
        )
        self.executor.evaluate(
            self.population,
            keyx="xNew",
            keyf="fNew",
            timingLabel="evaluate",
            timer=timer,
            target=self.target,
        )
        self.executor.foreach(
            self.population,
            self.accept,
            {
                "theta": evalf(self.theta, self.env["time"], self.rng),
                "goal": self.goal,
                "rng": self.rng,
            },
            timingLabel="accept",
            timer=timer,
        )

    @staticmethod
    def accept(
        ind: Individual, theta: float, goal: Goal, rng: random.Generator
    ) -> None:
        df = abs(ind["fNew"] - ind["f"])
        if goal.isBetter(ind["fNew"], ind["f"]) or rng.random() < exp(-df / theta):
            ind["f"] = ind["fNew"]
            ind["x"] = ind["xNew"].copy()
