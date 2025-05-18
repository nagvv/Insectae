import functools as ft
from itertools import chain, repeat, count
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# from .executor import BaseExecutor
from .targets import Target
from .timer import timing
from .typing import Environment, FuncKWArgs, Individual


@timing
def evaluate(
    population: List[Individual],
    keyx: str,
    keyf: str,
    target: Target,
    reEvalKey: Optional[str] = None,
    executor=None,
) -> None:
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


# must be in global scope to be able to be pickleable (i.e. sendable to workers)
def _call_wrap(
    op: Callable[..., None],
    obj: Any,
    fnpargs: Optional[Tuple] = None,
    fnkwargs: Optional[FuncKWArgs] = None,
):
    if fnpargs is None:
        fnpargs = ()
    if fnkwargs is None:
        fnkwargs = {}
    # with map/starmap interface we only able to send positional arguments,
    # so passing kwargs as positional argument and then unpacking it
    op(obj, *fnpargs, **fnkwargs)
    return obj


@timing
def foreach(
    population: List[Individual],
    op: Callable[..., None],
    fnkwargs: FuncKWArgs,
    executor=None,
) -> None:
    if executor is None:
        for ind in population:
            op(ind, **fnkwargs)
        return

    population[:] = executor.starmap(
        _call_wrap, zip(repeat(op), population, repeat(None), repeat(fnkwargs))
    )


@timing
def neighbors(
    population: List[Individual],
    op: Callable[..., None],
    permutation: List[int],
    fnkwargs: FuncKWArgs,
    executor=None,
) -> None:
    if executor is None:
        for i in range(len(permutation) // 2):
            inds_pair = (
                population[permutation[2 * i]],
                population[permutation[2 * i + 1]],
            )
            op(inds_pair, twoway=True, **fnkwargs)
        return

    shuffled = [population[i] for i in permutation]
    shuffled[:] = chain(
        *executor.starmap(
            _call_wrap,
            zip(
                repeat(op),
                zip(shuffled[::2], shuffled[1::2]),
                repeat(None),
                repeat(fnkwargs | {"twoway": True}),
            ),
        )
    )
    for new_ind, idx in zip(shuffled, permutation):
        population[idx] = new_ind


@timing
def pairs(
    population1: List[Individual],
    population2: List[Individual],
    op: Callable[..., None],
    fnkwargs: FuncKWArgs,
    executor=None,
) -> None:
    if executor is None:
        for inds_pair in zip(population1, population2):
            op(inds_pair, twoway=False, **fnkwargs)
        return

    population1[:] = (
        ind
        for (ind, _) in executor.starmap(
            _call_wrap,
            zip(
                repeat(op),
                zip(population1, population2),
                repeat(None),
                repeat(fnkwargs | {"twoway": False}),
            ),
        )
    )


@timing
def pop2ind(
    population1: List[Individual],
    population2: List[Individual],
    op: Callable[..., None],
    fnkwargs: FuncKWArgs,
    executor=None,
) -> None:
    if executor is None:
        for idx in range(len(population1)):
            ind = population1[idx]
            op(ind, population2, index=idx, **fnkwargs)
        return

    population1[:] = executor.starmap(
        _call_wrap,
        zip(
            repeat(op),
            population1,
            repeat((population2,)),
            (fnkwargs | {"index": i} for i in count()),
        ),
    )


@timing
def reducePop(
    population: List[Individual],
    extract: Callable[[Individual], Any],
    reduce: Callable[[Any, Any], Any],
    initVal: Any = None,
    executor=None,
) -> Any:
    if initVal is not None:
        return ft.reduce(reduce, map(extract, population), initVal)
    return ft.reduce(reduce, map(extract, population))


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
