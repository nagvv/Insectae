from typing import Tuple

import numpy as np

from .alg_base import Algorithm
from .common import Shuffled, evalf
from .goals import Goal
from .targets import RandomRealVector, RealTarget
from .typing import Environment, Evaluable, Individual


class CompetitiveSwarmOptimizer(Algorithm):
    def __init__(
        self,
        socialFactor: Evaluable[float],
        delta: float,
        target: RealTarget,
        **kwargs,
    ) -> None:
        self.socialFactor = socialFactor
        self.delta = delta
        self.compete = Shuffled(self.tournament)
        super().__init__(target=target, **kwargs)
        self.target: RealTarget  # hint for type checkers, FIXME is it needed?

    @staticmethod
    def tournament(
        pair: Tuple[Individual, Individual],
        socialFactor: Evaluable[float],
        target: RealTarget,
        goal: Goal,
        env: Environment,
        twoway: bool,  # unused
    ) -> None:
        ind1, ind2 = pair
        dim = target.dimension
        phi = evalf(socialFactor, inds=[ind1, ind2], env=env)
        x = env["x"]

        if goal.isBetter(ind1["f"], ind2["f"]):
            winner, loser = ind1, ind2
        else:
            winner, loser = ind2, ind1
        winner["reEval"], loser["reEval"] = False, True
        rnd = np.random.rand(3, dim)
        loser["v"] = np.multiply(rnd[0], loser["v"])
        loser["v"] += np.multiply(rnd[1], winner["x"] - loser["x"])
        loser["v"] += np.multiply(rnd[2], phi * (x - loser["x"]))
        loser["x"] += loser["v"]

    def start(self) -> None:
        super().init_attributes("socialFactor x", "&x *f v reEval")
        self.executor.foreach(
            self.population,
            self.opInit,
            {"target": self.target, "key": "x", "env": self.env},
        )
        self.executor.evaluate(self.population, keyx="x", keyf="f", env=self.env)
        vel = self.delta * (self.target.bounds[1] - self.target.bounds[0])
        self.executor.foreach(
            self.population,
            RandomRealVector((-vel, vel)),
            {"target": self.target, "key": "v"},
        )

    def runGeneration(self) -> None:
        timer = self.env.get("timer")
        ext, post = lambda x: x["x"], lambda x: x / self.popSize
        self.env["x"] = self.executor.reducePop(
            population=self.population,
            extract=ext,
            op=np.add,
            post=post,
            timingLabel="reduce",
            timer=timer,
        )
        self.compete(
            self.population,
            socialFactor=self.socialFactor,
            target=self.target,
            goal=self.goal,
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
            env=self.env,
        )
