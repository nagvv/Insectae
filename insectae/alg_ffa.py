from typing import Callable, List, Optional

import numpy as np

from .alg_base import Algorithm
from .operators import FillAttribute, copyAttribute, simpleMove
from .goals import Goal
from .targets import Target
from .typing import Environment, Individual


class FireflyAlgorithm(Algorithm):
    def __init__(
        self,
        alpha: float,  # randomization coefficient
        betamin: float,  # attractiveness at zero distance
        gamma: float,  # light absorption coefficient
        theta: float,  # randomness reduction factor
        alphabest: Optional[float] = None,  # randomization coefficient for best
        opLimitVel: Callable[..., None] = None,
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
        self.executor.foreach(
            self.population,
            self.opInit,
            {"target": self.target, "key": "x", "env": self.env},
        )
        self.executor.evaluate(self.population, keyx="x", keyf="val", env=self.env)
        self.executor.foreach(
            self.population,
            FillAttribute(np.zeros((self.popSize, 1 + self.target.dimension))),
            {"key": "neighbors", "env": self.env},
        )
        self.executor.foreach(
            self.population,
            FillAttribute(np.zeros(self.target.dimension)),
            {"key": "vel", "env": self.env},
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
        env: Environment,
    ) -> None:
        rng = env["rng"]
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
                ) + rng.uniform(-0.5 * alpha, 0.5 * alpha, tx.shape)
                tx += tvel
                tval = target(tx, tval, True)
        if amBest:
            tx += rng.uniform(-0.5 * alphabest, 0.5 * alphabest, tx.shape)
            tval = target(tx, tval, True)
        ind["vel"] = tx - ind["x"]
        ind["tval"] = tval
        # FIXME random greatly slowes execution

    def runGeneration(self) -> None:
        assert isinstance(self.alphabest, float)
        self.alpha *= self.theta
        self.alphabest *= self.theta
        timer = self.env.get("timer")
        self.executor.pop2ind(
            self.population,
            self.population,
            self.op_p2i,
            timingLabel="op_p2i",
        )
        self.executor.foreach(
            self.population,
            self.updateVel,
            {
                "alpha": self.alpha,
                "betamin": self.betamin,
                "gamma": self.gamma,
                "alphabest": self.alphabest,
                "target": self.target,
                "goal": self.goal,
                "env": self.env,
            },
            timingLabel="updatevel",
            timer=timer,
        )
        if self.opLimitVel is not None:
            self.executor.foreach(
                self.population,
                self.opLimitVel,
                {
                    "key": "vel",
                    "env": self.env,
                },
                timingLabel="limitvel",
                timer=timer,
            )
        self.executor.foreach(
            self.population,
            simpleMove,
            {
                "keyx": "x",
                "keyv": "vel",
                "dt": 1.0,
            },
            timingLabel="move",
            timer=timer,
        )
        self.executor.foreach(
            self.population,
            copyAttribute,
            {
                "keyFrom": "tval",
                "keyTo": "val",
            },
            timingLabel="copytval",
            timer=timer,
        )
