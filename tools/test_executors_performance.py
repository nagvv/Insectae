from argparse import ArgumentParser
from functools import partial
from math import isclose, sqrt
from os import cpu_count
from time import perf_counter, perf_counter_ns
from typing import Any, Callable, Iterable, List

import numpy as np
from insectae import BaseExecutor
from insectae.parallel import (DaskExecutor, MPIExecutor,
                               MultiprocessingExecutor, ThreadingExecutor)
from insectae.typing import Individual
from numba import float64, njit
from numpy.typing import NDArray


def get_cpu_count() -> int:
    logical_cpu_count = cpu_count()
    if logical_cpu_count is None:
        print("warning: failed to determine the number of cpu cores")
        return 1
    # mpi may fail to spawn the requested number of processes if it exeeds
    # the amount of available physical core
    # in the general case there are 2 logical cores per 1 physical one
    # "-1" to exclude the master process
    return max(logical_cpu_count // 2 - 1, 1)


def measure_execution_time(func: Callable[..., Any], *args, **kwargs):
    """measure function execution time in seconds"""
    start_time = perf_counter()
    func(*args, **kwargs)
    return perf_counter() - start_time


def payload_func(x: NDArray):
    accum = 0
    for v1 in x:
        accum += v1 * v1 / sqrt(v1 + v1)
    return accum


@njit(float64(float64[:]), nogil=True)
def payload_func_nogil(x: NDArray):
    accum = 0
    for v1 in x:
        accum += v1 * v1 / sqrt(v1 + v1)
    return accum


def base_test_func(x: NDArray, dur_ms: int, payload_func: Callable[[NDArray], float]):
    dur_ns = dur_ms * 1_000_000
    start_time_ns = perf_counter_ns()
    while True:
        accum = payload_func(x)
        if (perf_counter_ns() - start_time_ns) >= dur_ns:
            break
    return accum


# a dummy target class that does the bare minimum
class DummyTarget:
    def __init__(self, target: Callable[[NDArray], float]) -> None:
        self._target = target

    def __call__(self, x: NDArray, f: float, reEval: bool) -> Any:
        return self._target(x)

    def get_func(self) -> Callable[[Any], Any]:
        return self._target

    def update(self, x: Any, f: Any, reEval: bool) -> None:
        pass


def run_tests(
    target_func: Callable[[NDArray], float],
    func_name: str,
    population: List[Individual],
    executor_gens: Iterable[Callable[[], BaseExecutor]],
    duration: int,
):
    # run functions once to let everything necessary to be created/cached
    expected_value = target_func(test_array)
    print(
        f"configured target function execution time ({func_name}):",
        measure_execution_time(target_func, test_array),
        "sec",
    )
    env = {"target": DummyTarget(target_func)}
    for executor_gen in executor_gens:
        executor = executor_gen()
        executor_name = type(executor).__name__
        # run evaluation once to let everything necessary to be created/cached
        executor.evaluate(population, keyx="x", keyf="f", env=env)
        evals = 0
        start_time = perf_counter()
        while True:
            executor.evaluate(population, keyx="x", keyf="f", env=env)
            evals += 1
            end_time = perf_counter()
            if end_time - start_time >= duration:
                break
        print(executor_name, func_name, (end_time - start_time) / evals, "sec")
        for ind in population:
            if not isclose(ind["f"], expected_value):
                print(
                    f"warning: {executor_name} returned wrong result: "
                    f"expected: {expected_value}, got: {ind['f']}"
                )
                break
        if hasattr(executor, "close"):
            executor.close()
    print()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--duration", type=int, default=3, help="minimum duration of each test"
    )
    parser.add_argument(
        "-a",
        "--array-size",
        type=int,
        default=100,
        help="test array lenght or target's dimension",
    )
    parser.add_argument(
        "-p", "--popsize", type=int, default=100, help="test population size"
    )
    parser.add_argument(
        "-n",
        "--numproc",
        type=int,
        default=get_cpu_count(),
        help="number of workers/processes/threads to use",
    )
    parser.add_argument(
        "-t",
        "--target-dur",
        type=int,
        default=10,
        help="how much time takes one execution of the target function in ms",
    )
    args = parser.parse_args()
    print(args)
    print()

    test_array = np.ones(args.array_size, dtype=np.float64)
    population = [{"x": test_array, "f": 0}] * args.popsize
    executor_gens = (
        partial(BaseExecutor),
        partial(ThreadingExecutor, processes=args.numproc),
        partial(MultiprocessingExecutor, processes=args.numproc),
        partial(MPIExecutor, max_workers=args.numproc),
        partial(DaskExecutor, n_workers=args.numproc, threads_per_worker=1),
    )
    run_tests(
        partial(base_test_func, dur_ms=args.target_dur, payload_func=payload_func),
        func_name="gil variant",
        executor_gens=executor_gens,
        population=population,
        duration=args.duration,
    )
    run_tests(
        partial(
            base_test_func, dur_ms=args.target_dur, payload_func=payload_func_nogil
        ),
        func_name="nogil variant",
        executor_gens=executor_gens,
        population=population,
        duration=args.duration,
    )
