from typing import Callable

from .alg_base import Algorithm


class GeneticAlgorithm(Algorithm):
    def __init__(
        self,
        opSelect: Callable[..., None],
        opCrossover: Callable[..., None],
        opMutate: Callable[..., None],
        **kwargs,
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
            population=self.population, keyx="x", keyf="f", target=self.target
        )

    def runGeneration(self) -> None:
        timer = self.env.get("timer")
        self.opSelect(
            self.population,
            key="f",
            env=self.env,
            timingLabel="select",
            timer=timer,
        )
        self.opCrossover(
            self.population,
            key="x",
            env=self.env,
            timingLabel="cross",
            timer=timer,
        )
        self.executor.foreach(
            self.population,
            self.opMutate,
            {"key": "x", "time": self.env["time"], "rng": self.rng},
            timingLabel="mutate",
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
