from functools import partial
from operator import itemgetter
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .alg_base import Algorithm
from .common import evalf
from .targets import BinaryTarget
from .typing import Environment, Evaluable, Individual


class UnivariateMarginalDistributionAlgorithm(Algorithm):
    def __init__(self, opSelect: Callable[..., None], **kwargs) -> None:
        self.opSelect = opSelect
        super().__init__(**kwargs)
        self.target: BinaryTarget

    def start(self) -> None:
        super().init_attributes("", "&x *f")
        self.executor.foreach(
            self.population,
            self.opInit,
            {"target": self.target, "key": "x", "env": self.env},
        )
        self.executor.evaluate(
            population=self.population, keyx="x", keyf="f", target=self.target
        )

    @staticmethod
    def generate(ind: Individual, probs: NDArray, env: Environment):
        gen = env["rng"].uniform(size=probs.shape)
        ind["x"] = (gen < probs).astype(int)

    @staticmethod
    def _extract(ind: Individual):
        return ind["x"]

    def runGeneration(self) -> None:
        timer = self.env.get("timer")
        self.opSelect(
            self.population,
            key="f",
            env=self.env,
            timingLabel="select",
            timer=timer,
        )
        probs = (
            self.executor.reducePop(
                population=self.population,
                extract=self._extract,
                reduce=np.add,
                timingLabel="reduce",
                timer=timer,
            )
            / self.popSize
        )
        self.executor.foreach(
            self.population,
            self.generate,
            {"probs": probs, "env": self.env},
            timingLabel="generate",
            timer=timer,
        )
        self.executor.evaluate(
            self.population,
            keyx="x",
            keyf="f",
            timingLabel="evaluate",
            timer=timer,
            target=self.target,
        )


class PopulationBasedIncrementalLearning(Algorithm):
    def __init__(
        self,
        probMutate: Callable[..., None],
        n_best: Evaluable[int],
        n_worst: Evaluable[int],
        p_max: Evaluable[float],
        p_min: Evaluable[float],
        learning_rate: Evaluable[float],
        **kwargs,
    ) -> None:
        self.probMutate = probMutate
        self._n_best = n_best
        self._n_worst = n_worst
        self._p_max = p_max
        self._p_min = p_min
        self._learning_rate = learning_rate
        super().__init__(**kwargs)
        self.target: BinaryTarget

    def start(self) -> None:
        super().init_attributes("p", "&x *f")
        self.executor.foreach(
            self.population,
            self.opInit,
            {"target": self.target, "key": "x", "env": self.env},
        )
        self.executor.evaluate(
            population=self.population, keyx="x", keyf="f", target=self.target
        )
        self.env["p"] = np.repeat(0.5, repeats=self.target.dimension)

    @staticmethod
    def generate(ind: Individual, probs: NDArray, env: Environment):
        gen = env["rng"].uniform(size=probs.shape)
        ind["x"] = (gen < probs).astype(int)

    @staticmethod
    def _extract(ind: Individual):
        return ind["x"]

    @staticmethod
    def _reduce_plus(p, x, learning_rate):
        return p + learning_rate * (x - p)

    @staticmethod
    def _reduce_minus(p, x, learning_rate):
        return p - learning_rate * (x - p)

    def runGeneration(self) -> None:
        timer = self.env.get("timer")
        self.population.sort(key=self.goal.get_cmp_to_key(itemgetter("f")))
        n_best = evalf(self._n_best, self.env["time"], self.rng)
        learning_rate = evalf(self._learning_rate, self.env["time"], self.rng)
        probs = self.env["p"]
        probs = self.executor.reducePop(
            population=self.population[:n_best],
            extract=self._extract,
            reduce=partial(self._reduce_plus, learning_rate=learning_rate),
            initVal=probs,
            timingLabel="reduce(probs)",
            timer=timer,
        )
        n_worst = evalf(self._n_worst, self.env["time"], self.rng)
        probs = self.executor.reducePop(
            population=self.population[self.popSize - n_worst :],
            extract=self._extract,
            reduce=partial(self._reduce_minus, learning_rate=learning_rate),
            initVal=probs,
            timingLabel="reduce(probs)",
            timer=timer,
        )
        self.probMutate({"": probs}, key="", time=self.env["time"], rng=self.rng)
        p_max = evalf(self._p_max, self.env["time"], self.rng)
        p_min = evalf(self._p_min, self.env["time"], self.rng)
        probs = probs.clip(min=p_min, max=p_max)
        self.env["p"] = probs
        self.executor.foreach(
            self.population,
            self.generate,
            {"probs": probs, "env": self.env},
            timingLabel="generate",
            timer=timer,
        )
        self.executor.evaluate(
            self.population,
            keyx="x",
            keyf="f",
            timingLabel="evaluate",
            timer=timer,
            target=self.target,
        )
