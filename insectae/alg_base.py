from typing import Any, Callable, Dict, List, Optional, Union

from .goals import Goal, ToMin, getGoal
from .targets import Target


class Algorithm:
    def __init__(
        self,
        target: Target,
        popSize: int,
        goal: Union[Goal, str] = ToMin(),
        stop: Callable[[Dict[str, Any]], bool] = lambda x: False,
        opInit: Optional[Callable[..., None]] = None,
        env: Dict[str, Any] = dict(),
    ) -> None:
        self.target = target
        self.goal = goal if isinstance(goal, Goal) else getGoal(goal)
        self.stop = stop
        self.popSize = popSize
        self.opInit = opInit if opInit is not None else target.defaultInit()
        self.env = env

        self.population: List[Dict[str, Any]] = []  # FIXME, from argument?
        self.additionalProcedures: Dict[str, List[Callable[..., None]]] = {
            "start": [],
            "enter": [],
            "exit": [],
            "finish": [],
        }
        self.decorators: List[str] = []

    def addProcedure(self, key: str, proc: Callable[..., None]) -> None:
        self.additionalProcedures[key].append(proc)

    def runAdds(self, key: str) -> None:
        for proc in self.additionalProcedures[key]:
            proc(self.population, self.env) # FIXME **self.env??

    def checkKey(self, key: str) -> str:  # FIXME make internal?
        if key[0] in "*&":
            d = {"*": "_f", "&": "_x"}
            self.env[d[key[0]]] = key[1:]
            return key[1:]
        return key

    def run(self) -> None:
        self.start()
        while not self.stop(self.env):
            self.enter()
            self.runGeneration()
            self.exit()
        self.finish()

    def start(self, envAttrs: str = "", indAttrs: str = "") -> None:
        # environment
        keys = ["target", "goal", "time", "popSize"] + envAttrs.split()
        self.env = dict((key, None) for key in keys)
        for key in keys:
            if key in self.__dict__:
                self.env[key] = self.__dict__[key]
        self.env["time"] = 0

        # population
        keys = list(map(self.checkKey, indAttrs.split()))
        ind = dict((key, None) for key in keys)
        self.population = [ind.copy() for _ in range(self.popSize)]
        self.runAdds("start")

    def enter(self) -> None:
        self.env["time"] += 1
        self.runAdds("enter")

    def runGeneration(self) -> None:
        pass

    def exit(self) -> None:
        self.runAdds("exit")

    def finish(self) -> None:
        self.runAdds("finish")
