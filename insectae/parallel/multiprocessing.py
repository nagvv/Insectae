from functools import reduce as ftreduce
from itertools import pairwise, repeat
from multiprocessing import Value
from multiprocessing.pool import Pool
from multiprocessing.sharedctypes import Synchronized
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from ..executor import BaseExecutor
from ..patterns import (allNeighbors, evaluate, foreach, neighbors, pairs,
                        pop2ind)
from ..targets import Target
from ..timer import timing
from ..typing import FuncKWArgs, Individual


class _ExecutorWithContext:
    def __init__(self, pool, chunksize: int) -> None:
        self.pool = pool
        self.chunksize = chunksize

    @staticmethod
    def execute_with_context(fn: Callable[..., Any], args: Tuple) -> Any:
        # matches the signature of _call_wrap in patterns.py
        _, _, _, fnkwargs = args
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
            chunksize=self.chunksize,
        )


class _EvaluateWithContext:
    def __init__(self, pool, chunksize: int) -> None:
        self.pool = pool
        self.chunksize = chunksize

    @staticmethod
    def evaluate_with_context(args: Tuple) -> Any:
        return worker_context["target"].get_func()(args)

    def starmap(
        self, _: Callable[..., Any], fnargs: Iterable[Tuple], **kwargs
    ) -> Iterable[Any]:
        assert self.pool is not None
        return self.pool.starmap(
            self.evaluate_with_context,
            fnargs,
            chunksize=self.chunksize,
        )


class MultiprocessingExecutor(BaseExecutor):
    def __init__(
        self, processes: int, chunksize: int = 1, patterns: Optional[Set[str]] = None
    ) -> None:
        super().__init__(patterns=patterns)
        self.processes = processes
        self.chunksize = chunksize
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
        self.context_keys = list(context.keys()) + ["rng"]

    def starmap(
        self, fn: Callable[..., Any], fnargs: Iterable[Tuple], **kwargs
    ) -> Iterable[Any]:
        assert self.pool is not None
        return self.pool.starmap(fn, fnargs, chunksize=self.chunksize)

    def evaluate(
        self,
        population: List[Individual],
        keyx: str,
        keyf: str,
        target: Target,
        reEvalKey: Optional[str] = None,
        **kwargs,
    ) -> None:
        if "evaluate" not in self.patterns:
            return evaluate(
                population, keyx, keyf, target, reEvalKey, executor=None, **kwargs
            )

        return evaluate(
            population,
            keyx,
            keyf,
            target,
            reEvalKey,
            executor=_EvaluateWithContext(self.pool, self.chunksize),
            **kwargs,
        )

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
            population,
            op,
            fnkwargs,
            executor=_ExecutorWithContext(self.pool, self.chunksize),
            **kwargs,
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
            executor=_ExecutorWithContext(self.pool, self.chunksize),
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
            executor=_ExecutorWithContext(self.pool, self.chunksize),
            **kwargs,
        )

    def pop2ind(
        self,
        population1: List[Individual],
        population2: List[Individual],
        op: Callable[..., None],
        fnkwargs: FuncKWArgs,
        **kwargs,
    ) -> None:
        if "pop2ind" not in self.patterns or "env" in fnkwargs:
            return pop2ind(
                population1, population2, op, fnkwargs, executor=None, **kwargs
            )

        for key in self.context_keys:
            if key in fnkwargs:
                fnkwargs[key] = None

        return pop2ind(
            population1,
            population2,
            op,
            fnkwargs,
            executor=_ExecutorWithContext(self.pool, self.chunksize),
            **kwargs,
        )

    @staticmethod
    def _extract_reduce(
        extract: Callable[[Individual], Any],
        reduce: Callable[[Any, Any], Any],
        population: List[Individual],
    ):
        return ftreduce(reduce, map(extract, population))

    @timing
    def reducePop(
        self,
        population: List[Individual],
        extract: Callable[[Individual], Any],
        reduce: Callable[[Any, Any], Any],
        initVal: Any = None,
    ) -> Any:
        if len(population) == 0 and initVal is None:
            raise TypeError("reduction on empty iterable with no initial value")
        elif len(population) == 0:  # initVal is provided
            return initVal

        num_batches = min(len(population), self.processes)
        batch_size = len(population) // num_batches
        remainder = len(population) % num_batches
        ranges = [
            batch_size * idx + min(remainder, idx) for idx in range(num_batches)
        ] + [len(population)]
        assert self.pool is not None
        intermediate_result = self.pool.starmap(
            self._extract_reduce,
            zip(
                repeat(extract),
                repeat(reduce),
                (population[begin:end] for begin, end in pairwise(ranges)),
            ),
            chunksize=1,  # the amount of jobs is equal to pool size
        )
        if initVal is not None:
            return ftreduce(reduce, intermediate_result, initVal)
        else:
            return ftreduce(reduce, intermediate_result)

    def allNeighbors(
        self,
        population: List[Individual],
        op: Callable[..., Any],
        op_fnkwargs: FuncKWArgs,
        post: Optional[Callable[..., Any]],
        post_fnkwargs: FuncKWArgs,
        **kwargs,
    ) -> None:
        if (
            "allNeighbors" not in self.patterns
            or "env" in op_fnkwargs
            or "env" in post_fnkwargs
        ):
            return allNeighbors(
                population,
                op,
                op_fnkwargs,
                post,
                post_fnkwargs,
                executor=None,
                **kwargs,
            )

        for _key in self.context_keys:
            if _key in op_fnkwargs:
                op_fnkwargs[_key] = None

        for _key in self.context_keys:
            if _key in post_fnkwargs:
                post_fnkwargs[_key] = None

        return allNeighbors(
            population,
            op,
            op_fnkwargs,
            post,
            post_fnkwargs,
            executor=_ExecutorWithContext(self.pool, self.chunksize),
            **kwargs,
        )
