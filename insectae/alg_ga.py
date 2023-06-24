from typing import Callable

from .alg_base import Algorithm


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
        super().init_attributes("", "&x *f")
        self.executor.foreach(
            self.population,
            self.opInit,
            {"target": self.target, "key": "x", "env": self.env},
        )
        self.executor.evaluate(
            population=self.population, keyx="x", keyf="f", env=self.env
        )

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
        self.opCrossover(
            self.population,
            key="x",
            target=self.target,
            timingLabel="cross",
            timer=timer,
            env=self.env,
        )
        self.executor.foreach(
            self.population,
            self.opMutate,
            {"key": "x", "env": self.env},
            timingLabel="mutate",
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
