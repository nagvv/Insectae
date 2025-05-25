from typing import Any, Tuple

import numpy as np

from .alg_base import Algorithm
from .common import evalf
from .goals import Goal
from .operators import ShuffledNeighbors
from .targets import RandomRealVector, RealTarget
from .typing import Evaluable, Individual


class CompetitiveSwarmOptimizer(Algorithm):
    def __init__(
        self,
        socialFactor: Evaluable[float],
        delta: float,
        target: RealTarget,
        **kwargs,
    ) -> None:
        super().__init__(target=target, **kwargs)
        self.socialFactor = socialFactor
        self.delta = delta
        self.compete = ShuffledNeighbors(op=self.tournament)
        self.target: RealTarget  # hint for type checkers, FIXME is it needed?

    @staticmethod
    def tournament(
        pair: Tuple[Individual, Individual],
        key: str,
        socialFactor: Evaluable[float],
        target: RealTarget,
        goal: Goal,
        avg_x: Any,
        rng: np.random.Generator,
        twoway: bool,
        time: int,
    ) -> None:
        assert twoway is True
        ind1, ind2 = pair
        dim = target.dimension
        phi = evalf(socialFactor, time, rng)

        if goal.isBetter(ind1[key], ind2[key]):
            winner, loser = ind1, ind2
        else:
            winner, loser = ind2, ind1
        winner["reEval"], loser["reEval"] = False, True
        rnd = rng.random(size=(3, dim))
        loser["v"] = np.multiply(rnd[0], loser["v"])
        loser["v"] += np.multiply(rnd[1], winner["x"] - loser["x"])
        loser["v"] += np.multiply(rnd[2], phi * (avg_x - loser["x"]))
        loser["x"] += loser["v"]

    def start(self) -> None:
        super().init_attributes("socialFactor x", "&x *f v reEval")
        self.executor.foreach(
            self.population,
            self.opInit,
            {"target": self.target, "key": "x", "env": self.env},
        )
        self.executor.evaluate(self.population, keyx="x", keyf="f", target=self.target)
        vel = self.delta * (self.target.bounds[1] - self.target.bounds[0])
        self.executor.foreach(
            self.population,
            RandomRealVector((-vel, vel)),
            {"target": self.target, "key": "v", "env": self.env},
        )

    @staticmethod
    def _extract(ind: Individual):
        return ind["x"]

    def runGeneration(self) -> None:
        timer = self.env.get("timer")
        self.env["avg_x"] = (
            self.executor.reducePop(
                population=self.population,
                extract=self._extract,
                reduce=np.add,
                timingLabel="reduce",
                timer=timer,
            )
            / self.popSize
        )
        self.compete(
            self.population,
            key="f",
            env=self.env,
            timingLabel="compete",
            timer=timer,
        )
        self.executor.evaluate(
            self.population,
            keyx="x",
            keyf="f",
            reEvalKey="reEval",
            timingLabel="evaluate",
            timer=timer,
            target=self.target,
        )
