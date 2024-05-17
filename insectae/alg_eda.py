from typing import Callable
import numpy as np
from operator import itemgetter
from numpy.typing import NDArray

from .alg_base import Algorithm
from .typing import Individual, Evaluable, Environment
from .targets import BinaryTarget
from .common import evalf


class UnivariateMarginalDistributionAlgorithm(Algorithm):
    def __init__(
        self,
        opSelect: Callable[..., None],
        **kwargs
    ) -> None:
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
            population=self.population, keyx="x", keyf="f", env=self.env
        )

    @staticmethod
    def generate(ind: Individual, probs: NDArray, env: Environment):
        gen = env["rng"].uniform(size=probs.shape)
        ind["x"] = (gen < probs).astype(int)

    def runGeneration(self) -> None:
        timer = self.env.get("timer")
        self.opSelect(
            self.population,
            key="f",
            goal=self.goal,
            timingLabel="select",
            timer=timer,
            env=self.env,
        )
        probs = self.executor.reducePop(
            population=self.population,
            extract=lambda ind: ind["x"],
            op=lambda a, b: a + b,
            post=lambda sum: sum / len(self.population),
            timingLabel="reduce",
            timer=timer,
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
            env=self.env,
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
        **kwargs
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
            population=self.population, keyx="x", keyf="f", env=self.env
        )
        self.env["p"] = np.repeat(0.5, repeats=self.target.dimension)

    @staticmethod
    def generate(ind: Individual, probs: NDArray, env: Environment):
        gen = env["rng"].uniform(size=probs.shape)
        ind["x"] = (gen < probs).astype(int)

    def runGeneration(self) -> None:
        timer = self.env.get("timer")
        self.population.sort(key=self.goal.get_cmp_to_key(itemgetter("f")))
        n_best = evalf(self._n_best, self.population, self.env)
        learning_rate = evalf(self._learning_rate, self.population, self.env)
        probs = self.env["p"]
        probs = self.executor.reducePop(
            population=self.population[:n_best],
            extract=lambda ind: ind["x"],
            op=lambda p, x: p + learning_rate * (x - p),
            post=lambda x: x,
            initVal=probs,
            timingLabel="reduce(probs)",
            timer=timer,
        )
        n_worst = evalf(self._n_worst, self.population, self.env)
        probs = self.executor.reducePop(
            population=self.population[self.popSize - n_worst:],
            extract=lambda ind: ind["x"],
            op=lambda p, x: p - learning_rate * (x - p),
            post=lambda x: x,
            initVal=probs,
            timingLabel="reduce(probs)",
            timer=timer,
        )
        self.probMutate({"": probs}, "", self.env)
        p_max = evalf(self._p_max, self.population, self.env)
        p_min = evalf(self._p_min, self.population, self.env)
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
            env=self.env,
        )

