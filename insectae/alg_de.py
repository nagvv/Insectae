from typing import Callable, List, Dict, Any

from .alg_base import Algorithm
from .common import evalf, samplex
from .goals import Goal
from .patterns import evaluate, foreach, pairs, pop2ind
from .typing import Evaluable, Individual


class DifferentialEvolution(Algorithm):
    def __init__(
        self,
        opMakeProbe: Callable[..., None],
        opCrossover: Callable[..., None],
        opSelect: Callable[..., None],
        **kwargs
    ) -> None:
        self.opMakeProbe = opMakeProbe
        self.opCrossover = opCrossover
        self.opSelect = opSelect
        self.probes: List[Dict[str, Any]] = []
        super().__init__(**kwargs)

    def start(self) -> None:
        super().init_attributes("", "&x *f")
        self.probes = [{"x": None, "f": None} for i in range(self.popSize)]
        foreach(self.population, self.opInit, key="x", **self.env)
        evaluate(self.population, keyx="x", keyf="f", env=self.env)

    def runGeneration(self) -> None:
        pop2ind(
            self.probes,
            self.population,
            self.opMakeProbe,
            keyx="x",
            keyf="f",
            timingLabel="makeprobes",
            **self.env
        )
        pairs(
            self.probes,
            self.population,
            self.opCrossover,
            key="x",
            timingLabel="crossover",
            **self.env
        )
        evaluate(self.probes, keyx="x", keyf="f", timingLabel="evaluate", env=self.env)
        pairs(
            self.population,
            self.probes,
            self.opSelect,
            key="f",
            timingLabel="select",
            **self.env
        )


def argbestDE(population: List[Individual], keyf: str, goal: Goal) -> int:
    if goal == "min":
        return min(enumerate(population), key=lambda x: x[1][keyf])[0]
    else:
        return max(enumerate(population), key=lambda x: x[1][keyf])[0]


class ProbeClassic:
    def __init__(self, weight: Evaluable[float]) -> None:
        self.weight = weight

    def __call__(
        self,
        ind: Individual,
        population: List[Individual],
        keyx: str,
        index: int,
        **kwargs
    ) -> None:
        weight = evalf(self.weight, inds=[ind], **kwargs)
        S = samplex(len(population), 3, [index])
        a, b, c = [population[i] for i in S]
        ind[keyx] = a[keyx] + weight * (b[keyx] - c[keyx])


class ProbeBest:
    def __init__(self, weight: Evaluable[float]) -> None:
        self.weight = weight

    def __call__(
        self,
        ind: Individual,
        population: List[Individual],
        keyx: str,
        index: int,
        **kwargs
    ) -> None:
        weight = evalf(self.weight, inds=[ind], **kwargs)
        i = argbestDE(population, kwargs["keyf"], kwargs["goal"])
        sample = [i] + samplex(len(population), 2, [index, i])
        a, b, c = [population[i] for i in sample]
        ind[keyx] = a[keyx] + weight * (b[keyx] - c[keyx])


class ProbeCur2Best:
    def __init__(self, weight: Evaluable[float]) -> None:
        self.weight = weight

    def __call__(
        self,
        ind: Individual,
        population: List[Individual],
        keyx: str,
        index: int,
        **kwargs
    ) -> None:
        weight = evalf(self.weight, inds=[ind], **kwargs)
        i = argbestDE(population, kwargs["keyf"], kwargs["goal"])
        S = [index, i] + samplex(len(population), 2, [index, i])
        cur, a, b, c = [population[i] for i in S]
        ind[keyx] = cur[keyx] + weight * (a[keyx] - cur[keyx] + b[keyx] - c[keyx])


class ProbeBest2:
    def __init__(self, weight: Evaluable[float]) -> None:
        self.weight = weight

    def __call__(
        self,
        ind: Individual,
        population: List[Individual],
        keyx: str,
        index: int,
        **kwargs
    ) -> None:
        weight = evalf(self.weight, inds=[ind], **kwargs)
        i = argbestDE(population, kwargs["keyf"], kwargs["goal"])
        S = [i] + samplex(len(population), 4, [index, i])
        a, b, c, d, e = [population[i] for i in S]
        ind[keyx] = a[keyx] + weight * (b[keyx] - c[keyx] + d[keyx] - e[keyx])


class probeRandom5:
    def __init__(self, weight: Evaluable[float]) -> None:
        self.weight = weight

    def __call__(
        self,
        ind: Individual,
        population: List[Individual],
        keyx: str,
        index: int,
        **kwargs
    ) -> None:
        weight = evalf(self.weight, inds=[ind], **kwargs)
        S = samplex(len(population), 5, [index])
        a, b, c, d, e = [population[i] for i in S]
        ind[keyx] = a[keyx] + weight * (b[keyx] - c[keyx] + d[keyx] - e[keyx])
