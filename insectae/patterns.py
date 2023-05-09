import functools as ft
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from .targets import Target
from .timer import timing
from .typing import Individual, Environment


@timing
def evaluate(
    population: List[Individual],
    keyx: str,
    keyf: str,
    env: Environment,
    reEvalKey: Optional[str] = None,
) -> None:
    # TODO this function can be parallelized by MPI for heavy target functions
    target: Target = env["target"]
    for ind in population:
        reEval = (reEvalKey is None) or ind[reEvalKey]
        ind[keyf] = target(x=ind[keyx], f=ind[keyf], reEval=reEval)


@timing
def foreach(population: List[Individual], op: Callable[..., None], **opkwargs) -> None:
    for ind in population:  # TODO parallel loop
        op(ind, **opkwargs)


@timing
def neighbors(
    population: List[Individual], op: Callable[..., None], permutation: List[int], **opkwargs
) -> None:
    for i in range(len(permutation) // 2):  # TODO parallel loop
        inds_pair = population[permutation[2 * i]], population[permutation[2 * i + 1]]
        op(inds_pair, twoway=True, **opkwargs)


@timing
def pairs(
    population1: List[Individual],
    population2: List[Individual],
    op: Callable[..., None],
    **opkwargs
) -> None:
    for inds_pair in zip(population1, population2):  # TODO parallel loop
        op(inds_pair, twoway=False, **opkwargs)


@timing
def pop2ind(
    population1: List[Individual],
    population2: List[Individual],
    op: Callable[..., None],
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
    shape: Callable[..., float],
    reduce: Callable[[NDArray[np.float64]], Any],
    keyx: str,
    keys: str,
    env: Environment,
) -> None:
    n = len(population)
    D = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = metrics(population[i][keyx], population[j][keyx])
    for i in range(n):
        ind = population[i]
        S = np.zeros(n)
        for j in range(n):
            S[j] = shape(D[i][j], inds=[population[i], population[j]], **env)
        ind[keys] = reduce(S)
