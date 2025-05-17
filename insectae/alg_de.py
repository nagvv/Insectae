import numpy as np
from typing import Any, Callable, Dict, List

from .alg_base import Algorithm
from .common import evalf, samplex, get_args_from_env
from .goals import Goal
from .typing import Environment, Evaluable, Individual


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
        self.executor.foreach(
            self.population,
            self.opInit,
            {"target": self.target, "key": "x", "env": self.env},
        )
        self.executor.evaluate(self.population, keyx="x", keyf="f", env=self.env)

    def runGeneration(self) -> None:
        timer = self.env.get("timer")
        self.executor.pop2ind(
            self.probes,
            self.population,
            self.opMakeProbe,
            fnkwargs={
                "keyx": "x",
                "keyf": "f",
                "time": self.env["time"],
                "goal": self.goal,
                "rng": self.rng,
            },
            timingLabel="makeprobes",
            timer=timer,
        )
        self.executor.pairs(
            self.probes,
            self.population,
            self.opCrossover,
            fnkwargs={
                "key": "x",
                **get_args_from_env(self.opCrossover, self.env)
            },
            timingLabel="crossover",
            timer=timer,
        )
        self.executor.evaluate(
            self.probes,
            keyx="x",
            keyf="f",
            timingLabel="evaluate",
            timer=timer,
            env=self.env,
        )
        self.executor.pairs(
            self.population,
            self.probes,
            self.opSelect,
            fnkwargs={
                "key": "f",
                **get_args_from_env(self.opSelect, self.env)
            },
            timingLabel="select",
            timer=timer,
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
        index: int,
        keyx: str,
        keyf: str,
        time: int,
        goal: Goal,
        rng: np.random.Generator
    ) -> None:
        weight = evalf(self.weight, time)
        S = samplex(len(population), 3, [index], rng)
        a, b, c = [population[i] for i in S]
        ind[keyx] = a[keyx] + weight * (b[keyx] - c[keyx])


class ProbeBest:
    def __init__(self, weight: Evaluable[float]) -> None:
        self.weight = weight

    def __call__(
        self,
        ind: Individual,
        population: List[Individual],
        index: int,
        keyx: str,
        keyf: str,
        time: int,
        goal: Goal,
        rng: np.random.Generator
    ) -> None:
        weight = evalf(self.weight, time)
        i = argbestDE(population, keyf, goal)
        sample = [i] + samplex(len(population), 2, [index, i], rng)
        a, b, c = [population[i] for i in sample]
        ind[keyx] = a[keyx] + weight * (b[keyx] - c[keyx])


class ProbeCur2Best:
    def __init__(self, weight: Evaluable[float]) -> None:
        self.weight = weight

    def __call__(
        self,
        ind: Individual,
        population: List[Individual],
        index: int,
        keyx: str,
        keyf: str,
        time: int,
        goal: Goal,
        rng: np.random.Generator
    ) -> None:
        weight = evalf(self.weight, time)
        i = argbestDE(population, keyf, goal)
        S = [index, i] + samplex(len(population), 2, [index, i], rng)
        cur, a, b, c = [population[i] for i in S]
        ind[keyx] = cur[keyx] + weight * (a[keyx] - cur[keyx] + b[keyx] - c[keyx])


class ProbeBest2:
    def __init__(self, weight: Evaluable[float]) -> None:
        self.weight = weight

    def __call__(
        self,
        ind: Individual,
        population: List[Individual],
        index: int,
        keyx: str,
        keyf: str,
        time: int,
        goal: Goal,
        rng: np.random.Generator
    ) -> None:
        weight = evalf(self.weight, time)
        i = argbestDE(population, keyf, goal)
        S = [i] + samplex(len(population), 4, [index, i], rng)
        a, b, c, d, e = [population[i] for i in S]
        ind[keyx] = a[keyx] + weight * (b[keyx] - c[keyx] + d[keyx] - e[keyx])


class probeRandom5:
    def __init__(self, weight: Evaluable[float]) -> None:
        self.weight = weight

    def __call__(
        self,
        ind: Individual,
        population: List[Individual],
        index: int,
        keyx: str,
        keyf: str,
        time: int,
        goal: Goal,
        rng: np.random.Generator
    ) -> None:
        weight = evalf(self.weight, time)
        S = samplex(len(population), 5, [index], rng)
        a, b, c, d, e = [population[i] for i in S]
        ind[keyx] = a[keyx] + weight * (b[keyx] - c[keyx] + d[keyx] - e[keyx])
