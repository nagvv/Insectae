from typing import Callable, List, Optional

import numpy as np

from .alg_base import Algorithm
from .common import FillAttribute, copyAttribute, simpleMove
from .goals import Goal
from .patterns import evaluate, foreach, pop2ind
from .targets import Target
from .typing import Individual


class FireflyAlgorithm(Algorithm):
    def __init__(
        self,
        alpha: float,  # randomization coefficient
        betamin: float,  # attractiveness at zero distance
        gamma: float,  # light absorption coefficient
        theta: float,  # randomness reduction factor
        alphabest: Optional[float] = None,  # randomization coefficient for best
        opLimitVel: Callable[..., None] = lambda ind, **kwargs: None,
        **kwargs
    ) -> None:
        self.opLimitVel = opLimitVel
        self.alpha = alpha
        self.alphabest = alphabest
        self.betamin = betamin
        self.gamma = gamma
        self.theta = theta
        super().__init__(**kwargs)

    def start(self) -> None:
        super().init_attributes(
            "alpha alphabest betamin gamma theta", "&x *val vel tval neighbors"
        )
        if self.env["alphabest"] is None:
            self.env["alphabest"] = self.env["alpha"]
        foreach(self.population, self.opInit, key="x", **self.env)
        evaluate(self.population, keyx="x", keyf="val", env=self.env)
        foreach(
            self.population,
            FillAttribute(np.zeros((self.popSize, 1 + self.target.dimension))),
            key="neighbors",
            **self.env
        )
        foreach(
            self.population,
            FillAttribute(np.zeros(self.target.dimension)),
            key="vel",
            **self.env
        )

    def op_p2i(
        self, ind: Individual, pop: List[Individual], index: int, **kwargs
    ) -> None:
        for idx in range(self.popSize):
            ind["neighbors"][idx, 0] = pop[idx]["val"]
            ind["neighbors"][idx, 1:] = pop[idx]["x"]

    @staticmethod
    def updateVel(
        ind: Individual,
        alpha: float,
        betamin: float,
        gamma: float,
        alphabest: float,
        target: Target,
        goal: Goal,
        **kwargs
    ) -> None:
        rand = np.random.default_rng()
        tval = ind["val"]
        tx = ind["x"].copy()
        amBest = True
        for nbfx in ind["neighbors"]:
            nbf = nbfx[0]
            nbx = nbfx[1:]
            if goal.isBetter(nbf, tval):
                amBest = False
                dist = np.linalg.norm(tx - nbx)
                tvel = betamin * np.exp(-gamma * (dist**2)) * (
                    nbx - tx
                ) + rand.uniform(-0.5 * alpha, 0.5 * alpha, tx.shape)
                tx += tvel
                tval = target(tx, tval, True)
        if amBest:
            tx += rand.uniform(-0.5 * alphabest, 0.5 * alphabest, tx.shape)
            tval = target(tx, tval, True)
        ind["vel"] = tx - ind["x"]
        ind["tval"] = tval
        # TODO random greatly slowes execution

    def runGeneration(self) -> None:
        assert isinstance(self.alphabest, float)
        self.alpha *= self.theta
        self.alphabest *= self.theta
        pop2ind(
            self.population,
            self.population,
            self.op_p2i,
            timingLabel="op_p2i",
            **self.env
        )
        foreach(self.population, self.updateVel, timingLabel="updatevel", **self.env)
        foreach(
            self.population,
            self.opLimitVel,
            key="vel",
            timingLabel="limitvel",
            **self.env
        )
        foreach(
            self.population,
            simpleMove,
            keyx="x",
            keyv="vel",
            dt=1.0,
            timingLabel="move",
        )
        foreach(
            self.population,
            copyAttribute,
            keyFrom="tval",
            keyTo="val",
            timingLabel="copytval",
        )
