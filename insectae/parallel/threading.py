from functools import reduce as ftreduce
from itertools import pairwise, repeat
from multiprocessing.pool import ThreadPool
from typing import Any, Callable, Iterable, List, Optional, Set, Tuple

from ..executor import BaseExecutor
from ..timer import timing
from ..typing import Individual


class ThreadingExecutor(BaseExecutor):
    def __init__(
        self,
        processes: int,
        chunksize: Optional[int] = None,
        patterns: Optional[Set[str]] = None,
    ) -> None:
        super().__init__(patterns=patterns)
        self.processes = processes
        self.chunksize = chunksize
        self.pool = ThreadPool(processes=self.processes)

    def starmap(
        self, fn: Callable[..., Any], fnargs: Iterable[Tuple], **kwargs
    ) -> Iterable[Any]:
        assert self.pool is not None
        return self.pool.starmap(fn, fnargs, chunksize=self.chunksize)

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
