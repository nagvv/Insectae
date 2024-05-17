from typing import Callable, List

import numpy as np
from numpy.typing import NDArray

from .alg_base import Algorithm
from .decorators import RankIt
from .goals import Goal
from .typing import Individual


class BeesAlgorithm(Algorithm):
    def __init__(
        self,
        beesNum: int,
        opLocal: Callable[..., None],
        opGlobal: Callable[..., None],
        opProbs: Callable[[int], NDArray[np.float64]],
        **kwargs
    ) -> None:
        self.beesNum = beesNum
        self.opLocal = opLocal
        self.opGlobal = opGlobal
        self.opProbs = opProbs
        self.opFly: Callable[..., None] = lambda: None
        self.bees: List[Individual] = []
        super().__init__(**kwargs)
        RankIt()(self)

    def start(self) -> None:
        super().init_attributes("", "&x *f")
        bee: Individual = {"x": None, "f": None, "_rank": None}
        self.bees = [bee.copy() for i in range(self.beesNum)]
        self.opFly = OpFly(self.opProbs, self.opLocal, self.opGlobal, self.popSize)
        self.executor.foreach(
            self.population,
            self.opInit,
            {"target": self.target, "key": "x", "env": self.env},
        )
        self.executor.evaluate(self.population, keyx="x", keyf="f", env=self.env)

    @staticmethod
    def updatePlace(
        place: Individual, bees: List[Individual], goal: Goal, **kwargs
    ) -> None:
        for bee in bees:
            if bee["_rank"] == place["_rank"] and goal.isBetter(bee["f"], place["f"]):
                place["f"] = bee["f"]
                place["x"] = bee["x"].copy()

    def runGeneration(self) -> None:
        self.executor.pop2ind(
            self.bees,
            self.population,
            self.opFly,
            key="x",
            timingLabel="fly",
            **self.env
        )
        self.executor.evaluate(
            self.bees,
            keyx="x",
            keyf="f",
            timingLabel="evaluate",
            timer=self.env.get("timer"),
            env=self.env,
        )
        self.executor.pop2ind(
            self.population,
            self.bees,
            self.updatePlace,
            timingLabel="update",
            **self.env
        )


class OpFly:
    def __init__(
        self,
        opProbs: Callable[[int], NDArray[np.float64]],
        opLocal: Callable[..., None],
        opGlobal: Callable[..., None],
        psize: int,
    ) -> None:
        self.opLocal = opLocal
        self.opGlobal = opGlobal
        self.probs = opProbs(psize)

    def __call__(self, bee, places, target, key, **kwargs) -> None:
        rand_val = kwargs["rng"].random()
        for place in places:
            prob = self.probs[place["_rank"]]
            if rand_val < prob:
                bee["_rank"] = place["_rank"]
                if place["_rank"] == len(places) - 1:
                    self.opGlobal(ind=bee, target=target, key=key, env=kwargs)
                else:
                    bee["x"] = place["x"].copy()
                    self.opLocal(ind=bee, key=key, env=kwargs)
                return
            rand_val -= prob


class UniformPlacesProbs:
    def __init__(self, pscout: float) -> None:
        self.pscout = pscout

    def __call__(self, size) -> NDArray[np.float64]:
        probs = np.full(size, (1.0 - self.pscout) / (size - 1.0))
        probs[size - 1] = self.pscout
        return probs


class LinearPlacesProbs:
    def __init__(self, elitism: float, pscout: float) -> None:
        self.elitism = elitism
        self.pscout = pscout

    def __call__(self, size: int) -> NDArray[np.float64]:
        a = self.elitism * (1.0 - self.pscout) * 2.0 / ((size - 1) * (size - 2))
        b = (1.0 - self.pscout) / (size - 1) + a * (size - 2) / 2.0
        probs = -a * np.array(range(size)) + b
        probs[size - 1] = self.pscout
        return probs


class BinaryPlacesProbs:
    def __init__(self, rho: float, elitism: float, pscout: float) -> None:
        self.rho = rho
        self.mu = 1.0 / (1.0 - elitism)
        self.pscout = pscout

    def __call__(self, size: int) -> NDArray[np.float64]:
        me = int((size - 1) * self.rho)
        mo = size - 1 - me
        pe = (1.0 - self.pscout) / (me + mo / self.mu)
        po = pe / self.mu
        return np.array([pe] * me + [po] * mo + [self.pscout])
