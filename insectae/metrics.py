from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt  # type: ignore
import numpy as np

from .goals import Goal, ToMin, getGoal


# FIXME: add BaseMetrics for user defined metrics? ABC?
class Metrics:
    def __init__(self, goal: Union[Goal, str] = ToMin(), verbose: int = 0) -> None:
        self.bestSolution: Any = None
        self.bestValue: Any = None
        self.efs: int = 0
        self.currentGeneration: int = 0
        self.data: List[List[Any]] = [[]]
        self.verbose: int = verbose
        if isinstance(goal, str):
            self.goal = getGoal(goal)
        else:
            self.goal = goal
        self.timing: Dict[str, float] = {}

    def newEval(self, x: Any, f: Any, reEval: bool) -> None:
        if reEval:
            self.efs += 1
            if self.bestValue is None or self.goal.isBetter(f, self.bestValue):
                self.bestValue = f
                self.bestSolution = np.copy(x)
        self.data[-1].append(f)

    def newGeneration(self) -> None:
        if self.verbose > 0 and self.currentGeneration % self.verbose == 0:
            g, e, b = self.currentGeneration, self.efs, self.bestValue
            print(f"Generation: {g}, EFs: {e}, target: {b}")
        self.currentGeneration += 1
        self.data.append([])

    def show(
        self,
        width: int = 8,
        height: int = 6,
        log: bool = False,
        bottom: Optional[int] = None,
    ) -> None:
        bests, averages, medians = [], [], []
        for rec in self.data:
            if len(rec) == 0:
                continue
            x = np.array(rec)
            if self.goal == "min":
                bests.append(np.min(x))
            else:
                bests.append(np.max(x))
            averages.append(np.mean(x))
            medians.append(np.median(x))
        x_axis = list(range(len(bests)))
        ax = plt.subplots(figsize=(width, height))[1]
        ax.plot(x_axis, bests, color="green", label="best")
        ax.plot(x_axis, averages, color="blue", label="average")
        ax.plot(x_axis, medians, color="red", label="median")
        ax.legend()
        if log:
            ax.set_yscale("log")
        bot, top = plt.ylim()
        if bottom is not None:
            bot = bottom
        plt.ylim(bot, top)
        plt.show()

    def _format(self, label, L, t, T) -> str:
        p = ""
        if T > 0:
            p = "({:.2f}%)".format(t / T * 100)
        lab = ("{:" + str(L) + "s} ").format(label)
        return "  " + lab + ": {:6.3f} ".format(t) + p

    def showTiming(self) -> None:
        print("[timing]")
        if len(self.timing) == 0:
            print("  no timing information")
            return
        labels = self.timing.keys()
        max_label_len = max(map(len, labels))
        total = None
        # save up total for later
        if "total" in labels:
            total = self.timing["total"]
        for label, value in self.timing.items():
            if label == "total":
                continue
            print(self._format(label, max_label_len, value, total))
        print("  " + "=" * max_label_len)
        # print total last if it exists
        if total is not None:
            print(self._format("total", max_label_len, total, 0))
