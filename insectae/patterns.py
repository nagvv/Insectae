import functools as ft
from typing import Any, Callable, List, Optional
from itertools import repeat

import numpy as np
from numpy.typing import NDArray

# from .executor import BaseExecutor
from .targets import Target
from .timer import timing
from .typing import Environment, Individual, FuncKWArgs


@timing
def evaluate(
    population: List[Individual],
    keyx: str,
    keyf: str,
    env: Environment,
    reEvalKey: Optional[str] = None,
    executor=None,  # FIXME circle dep?
) -> None:
    target: Target = env["target"]
    if executor is None:
        for ind in population:
            reEval = (reEvalKey is None) or ind[reEvalKey]
            ind[keyf] = target(x=ind[keyx], f=ind[keyf], reEval=reEval)
        return

    # find which individuals need to be evaluated
    to_evaluate: List[Individual] = []
    to_skip: List[Individual] = []
    for ind in population:
        reEval = (reEvalKey is None) or ind[reEvalKey]
        if reEval:
            to_evaluate.append(ind)
        else:
            to_skip.append(ind)
    # enqueue jobs
    results = executor.starmap(
        target.get_func(),
        ((ind[keyx],) for ind in to_evaluate),
    )
    # update metrics for non-evaluated ones
    for ind in to_skip:
        target.update(x=ind[keyx], f=ind[keyf], reEval=False)
    # update metrics with results
    for ind, new_f in zip(to_evaluate, results):
        ind[keyf] = new_f
        target.update(x=ind[keyx], f=new_f, reEval=True)


def _call_wrap(op: Callable[..., None], ind: Individual, fnkwargs: FuncKWArgs):
    # with map/starmap interface we only able to send positional arguments,
    # so passing kwargs as positional argument and then unpacking it
    op(ind, **fnkwargs)
    return ind


@timing
def foreach(
    population: List[Individual],
    op: Callable[..., None],
    fnkwargs: FuncKWArgs,
    executor=None,
) -> None:
    for ind in population:
        op(ind, **fnkwargs)
    # if executor is None:
    #     for ind in population:
    #         op(ind, **fnkwargs)
    #     return
    # # TODO ensure that fnkwargs is sent once per worker?
    # # FIXME: doesn't work properly for multiprocessing/mpi?
    # population[:] = executor.starmap(
    #     _call_wrap,
    #     zip(repeat(op), population, repeat(fnkwargs))
    # )


@timing
def neighbors(
    population: List[Individual],
    op: Callable[..., None],
    permutation: List[int],
    executor=None,
    **opkwargs
) -> None:
    for i in range(len(permutation) // 2):  # TODO parallel loop
        inds_pair = population[permutation[2 * i]], population[permutation[2 * i + 1]]
        op(inds_pair, twoway=True, **opkwargs)


@timing
def pairs(
    population1: List[Individual],
    population2: List[Individual],
    op: Callable[..., None],
    executor=None,
    **opkwargs
) -> None:
    for inds_pair in zip(population1, population2):  # TODO parallel loop
        op(inds_pair, twoway=False, **opkwargs)


@timing
def pop2ind(
    population1: List[Individual],
    population2: List[Individual],
    op: Callable[..., None],
    executor=None,
    **opkwargs
) -> None:
    for idx in range(len(population1)):  # parallel loop
        ind = population1[idx]
        op(ind, population2, index=idx, **opkwargs)


# reducing population into single value (can be parallelized as binary tree)
@timing
def reducePop(
    population: List[Individual],
    extract: Callable[[Individual], Any],
    op: Callable[[Any, Any], Any],
    post: Callable[[Any], Any],
    initVal: Any = None,
    executor=None,
) -> Any:
    if initVal is not None:
        return post(ft.reduce(op, map(extract, population), initVal))
    return post(ft.reduce(op, map(extract, population)))


# TODO allow non-number metric
# calculating distances between all individuals
# for each individual convert distances to signals and reduce them to single value
@timing
def signals(
    population: List[Individual],
    metrics: Callable[[Any, Any], float],
    shape: Callable[..., Any],
    reduce: Callable[[NDArray[np.float64]], Any],
    keyx: str,
    keys: str,
    env: Environment,
    executor=None,
) -> None:
    n = len(population)
    D = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = metrics(population[i][keyx], population[j][keyx])
    for i in range(n):
        ind = population[i]
        first_val = shape(D[i, 0], inds=[ind, population[0]], env=env)
        S = np.zeros((n, *np.shape(first_val)))
        S[0] = first_val
        for j in range(1, n):
            S[j] = shape(D[i, j], inds=[ind, population[j]], env=env)
        ind[keys] = reduce(S)
