from random import random
from typing import Callable

from numpy import exp

from .alg_base import Algorithm
from .common import copyAttribute, evalf
from .goals import Goal
from .patterns import evaluate, foreach
from .typing import Evaluable, Individual


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
        super().start("theta", "&x xNew *f fNew")
        foreach(self.population, self.opInit, key="x", **self.env)
        evaluate(self.population, keyx="x", keyf="f", env=self.env)

    def runGeneration(self) -> None:
        foreach(
            self.population,
            copyAttribute,
            keyFrom="x",
            keyTo="xNew",
            timingLabel="copy"
        )
        foreach(
            self.population, self.opMove, key="xNew", timingLabel="move", **self.env
        )
        evaluate(
            self.population,
            keyx="xNew",
            keyf="fNew",
            timingLabel="evaluate",
            env=self.env,
        )
        foreach(self.population, self.accept, timingLabel="accept", **self.env)

    @staticmethod
    def accept(ind: Individual, theta: Evaluable[float], goal: Goal, **kwargs) -> None:
        theta = evalf(theta, inds=[ind], **kwargs)
        df = abs(ind["fNew"] - ind["f"])
        if goal.isBetter(ind["fNew"], ind["f"]) or random() < exp(-df / theta):
            ind["f"] = ind["fNew"]
            ind["x"] = ind["xNew"].copy()
