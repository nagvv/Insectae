from typing import Tuple, Any

import numpy as np

from .alg_base import Algorithm
from .common import ShuffledNeighbors, evalf
from .goals import Goal
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
        self.compete = ShuffledNeighbors(op=self.tournament, rng=self.rng, executor=self.executor)
        self.target: RealTarget  # hint for type checkers, FIXME is it needed?

    @staticmethod
    def tournament(
        pair: Tuple[Individual, Individual],
        phi: float,
        target: RealTarget,
        goal: Goal,
        avg_x: Any,
        rng: np.random.Generator,
        twoway: bool,
    ) -> None:
        assert twoway is True
        ind1, ind2 = pair
        dim = target.dimension

        if goal.isBetter(ind1["f"], ind2["f"]):
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
        self.executor.evaluate(self.population, keyx="x", keyf="f", env=self.env)
        vel = self.delta * (self.target.bounds[1] - self.target.bounds[0])
        self.executor.foreach(
            self.population,
            RandomRealVector((-vel, vel)),
            {"target": self.target, "key": "v", "env": self.env},
        )

    def runGeneration(self) -> None:
        timer = self.env.get("timer")
        ext, post = lambda x: x["x"], lambda x: x / self.popSize
        self.env["avg_x"] = self.executor.reducePop(
            population=self.population,
            extract=ext,
            op=np.add,
            post=post,
            timingLabel="reduce",
            timer=timer,
        )
        self.compete(
            self.population,
            fnkwargs={
                "phi": evalf(self.socialFactor, self.env["time"]),
                "target": self.target,
                "goal": self.goal,
                "avg_x": self.env["avg_x"],
                "rng": self.rng,
            },
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
