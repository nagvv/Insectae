from typing import Callable, Dict, List, Optional, Sequence, Union

from numpy.random import Generator, default_rng

from .executor import BaseExecutor
from .goals import Goal, ToMin, getGoal
from .targets import Target
from .typing import Environment, Individual


class Algorithm:
    def __init__(
        self,
        target: Target,
        popSize: int,
        goal: Union[Goal, str] = ToMin(),
        stop: Callable[[Environment], bool] = lambda x: False,
        opInit: Optional[Callable[..., None]] = None,
        env: Optional[Environment] = None,
        executor: Optional[BaseExecutor] = None,
        rng_seed: Union[None, int, Sequence[int], Generator] = None,
    ) -> None:
        self.target = target
        self.goal = goal if isinstance(goal, Goal) else getGoal(goal)
        self.stop = stop
        self.popSize = popSize
        self.opInit = opInit if opInit is not None else target.defaultInit()
        self.env = env if env is not None else {}
        self.executor = executor if executor is not None else BaseExecutor()
        self.rng = default_rng(rng_seed)

        self.executor.init(
            context={"target": self.target, "goal": self.goal},
            rng=self.rng,
        )

        self.population: List[Individual] = []
        self.additionalProcedures: Dict[str, List[Callable[..., None]]] = {
            "start": [],
            "enter": [],
            "exit": [],
            "finish": [],
        }
        self.decorators: List[str] = []

    def addProcedure(self, key: str, proc: Callable[..., None]) -> None:
        self.additionalProcedures[key].append(proc)

    def runAdds(self, key: str, reverse: bool = False) -> None:
        procedures = self.additionalProcedures[key]
        if reverse:
            procedures = reversed(procedures)
        for proc in procedures:
            proc(self.population, self.env)

    def checkKey(self, key: str) -> str:
        if key[0] in "*&":
            keymap = {"*": "solutionValueLabel", "&": "solutionLabel"}
            self.env[keymap[key[0]]] = key[1:]
            return key[1:]
        return key

    def run(self) -> None:
        self.start()
        self.runAdds("start")
        while not self.stop(self.env):
            self.enter()
            self.runAdds("enter")
            self.runGeneration()
            self.exit()
            self.runAdds("exit", reverse=True)
        self.finish()
        self.runAdds("finish", reverse=True)

    def init_attributes(self, envAttrs: str, indAttrs: str) -> None:
        # environment
        keys = [
            "target",
            "goal",
            "time",
            "popSize",
            "rng",
            "executor",
        ] + envAttrs.split()
        self.env.update({key: None for key in keys})
        for key in keys:
            if key in self.__dict__:
                self.env[key] = self.__dict__[key]
        self.env["time"] = 0

        # population
        keys = list(map(self.checkKey, indAttrs.split()))
        ind = {key: None for key in keys}
        self.population = [ind.copy() for _ in range(self.popSize)]

    def start(self) -> None:
        self.init_attributes("", "")

    def enter(self) -> None:
        assert isinstance(self.env["time"], int)
        self.env["time"] += 1

    def runGeneration(self) -> None:
        pass

    def exit(self) -> None:
        pass

    def finish(self) -> None:
        pass
