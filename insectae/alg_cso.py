from typing import Any, Tuple

import numpy as np

from .alg_base import Algorithm
from .common import Shuffled, evalf
from .goals import Goal
from .patterns import evaluate, foreach, reducePop
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
        self.socialFactor = socialFactor
        self.delta = delta
        self.compete = Shuffled(self.tournament)
        super().__init__(target=target, **kwargs)
        self.target: RealTarget  # hint for type checkers, FIXME is it needed?

    @staticmethod
    def tournament(
        pair: Tuple[Individual, Individual],
        x: Any,
        socialFactor: Evaluable[float],
        target: RealTarget,
        goal: Goal,
        **kwargs,
    ) -> None:
        ind1, ind2 = pair
        dim = target.dimension
        phi = evalf(socialFactor, inds=[ind1, ind2], **kwargs)

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
        super().start("socialFactor x", "&x *f v reEval")
        foreach(self.population, self.opInit, key="x", **self.env)
        evaluate(self.population, keyx="x", keyf="f", env=self.env)
        vel = self.delta * (self.target.bounds[1] - self.target.bounds[0])
        foreach(self.population, RandomRealVector((-vel, vel)), key="v", **self.env)

    def runGeneration(self) -> None:
        ext, post = lambda x: x["x"], lambda x: x / self.popSize
        self.env["x"] = reducePop(
            population=self.population,
            extract=ext,
            op=np.add,
            post=post,
            timingLabel="reduce",
        )
        self.compete(self.population, timingLabel="compete", **self.env)
        evaluate(
            self.population,
            keyx="x",
            keyf="f",
            reEvalKey="reEval",
            timingLabel="evaluate",
            env=self.env,
        )
