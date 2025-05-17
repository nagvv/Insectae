from itertools import repeat
from multiprocessing import Value
from multiprocessing.pool import Pool
from multiprocessing.sharedctypes import Synchronized
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from ..executor import BaseExecutor
from ..patterns import foreach, neighbors, pairs
from ..typing import FuncKWArgs, Individual


class _ExecutorWithContext:
    def __init__(self, pool) -> None:
        self.pool = pool

    @staticmethod
    def execute_with_context(fn: Callable[..., Any], args: Tuple) -> Any:
        _, _, fnkwargs = args
        for key, item in worker_context.items():
            if key in fnkwargs:
                fnkwargs[key] = item
        return fn(*args)

    def starmap(
        self, fn: Callable[..., Any], fnargs: Iterable[Tuple], **kwargs
    ) -> Iterable[Any]:
        assert self.pool is not None
        return self.pool.starmap(
            self.execute_with_context,
            zip(repeat(fn), fnargs),
            chunksize=1,  # TODO add batching
        )


class MultiprocessingExecutor(BaseExecutor):
    def __init__(self, processes: int, patterns: Optional[Set[str]] = None) -> None:
        super().__init__(patterns=patterns)
        self.processes = processes
        self.pool = None

    @staticmethod
    def fill_globals(
        context: Dict[str, Any], rngs: List[np.random.Generator], counter: Synchronized
    ) -> None:
        global worker_context, worker_index
        worker_context = context
        with counter.get_lock():
            worker_index = counter.value
            counter.value += 1
        worker_context["rng"] = rngs[worker_index]

    def init(self, context: Dict[str, Any], rng: np.random.Generator) -> None:
        # FIXME: we can't guarantee rng reproducibility until we make starmap
        # to have a reproducible behavior
        counter = Value("i", 0)
        self.pool = Pool(
            processes=self.processes,
            initializer=self.fill_globals,
            initargs=(context, rng.spawn(self.processes), counter),
        )
        self.context_keys = list(context.keys())

    def starmap(
        self, fn: Callable[..., Any], fnargs: Iterable[Tuple], **kwargs
    ) -> Iterable[Any]:
        assert self.pool is not None
        return self.pool.starmap(fn, fnargs, chunksize=1)  # TODO add batching

    def foreach(
        self,
        population: List[Individual],
        op: Callable[..., None],
        fnkwargs: FuncKWArgs,
        **kwargs,
    ) -> None:
        # Currently, it is impossible/unsafe to send env to the workers as an
        # argument; so when see env being used as an argument fallback to
        # sequential implementation;
        # Note: this check work only when env argument is named exactly as "env"
        if "foreach" not in self.patterns or "env" in fnkwargs:
            return foreach(population, op, fnkwargs, executor=None, **kwargs)

        # target, goal and rng already exist in workers, so do not send them
        for key in self.context_keys:
            if key in fnkwargs:
                fnkwargs[key] = None

        return foreach(
            population, op, fnkwargs, executor=_ExecutorWithContext(self.pool), **kwargs
        )

    def neighbors(
        self,
        population: List[Individual],
        op: Callable[..., None],
        permutation: List[int],
        fnkwargs: FuncKWArgs,
        **kwargs,
    ) -> None:
        if "neighbors" not in self.patterns or "env" in fnkwargs:
            return neighbors(
                population, op, permutation, fnkwargs, executor=None, **kwargs
            )

        for key in self.context_keys:
            if key in fnkwargs:
                fnkwargs[key] = None

        return neighbors(
            population,
            op,
            permutation,
            fnkwargs,
            executor=_ExecutorWithContext(self.pool),
            **kwargs,
        )

    def pairs(
        self,
        population1: List[Individual],
        population2: List[Individual],
        op: Callable[..., None],
        fnkwargs: FuncKWArgs,
        **kwargs,
    ) -> None:
        if "pairs" not in self.patterns or "env" in fnkwargs:
            return pairs(
                population1, population2, op, fnkwargs, executor=None, **kwargs
            )

        for key in self.context_keys:
            if key in fnkwargs:
                fnkwargs[key] = None

        return pairs(
            population1,
            population2,
            op,
            fnkwargs,
            executor=_ExecutorWithContext(self.pool),
            **kwargs,
        )
