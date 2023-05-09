from typing import Callable

from .alg_base import Algorithm
from .patterns import evaluate, foreach


class GeneticAlgorithm(Algorithm):
    def __init__(
        self,
        opSelect: Callable[..., None],
        opCrossover: Callable[..., None],
        opMutate: Callable[..., None],
        **kwargs
    ) -> None:
        self.opSelect = opSelect
        self.opCrossover = opCrossover
        self.opMutate = opMutate
        super().__init__(**kwargs)

    def start(self) -> None:
        super().start("", "&x *f")
        foreach(population=self.population, op=self.opInit, key="x", **self.env)
        evaluate(population=self.population, keyx="x", keyf="f", env=self.env)

    def runGeneration(self) -> None:
        self.opSelect(self.population, key="f", timingLabel="select", **self.env)
        self.opCrossover(self.population, key="x", timingLabel="cross", **self.env)
        foreach(
            self.population, self.opMutate, key="x", timingLabel="mutate", **self.env
        )
        evaluate(
            self.population, keyx="x", keyf="f", timingLabel="evaluate", env=self.env
        )
