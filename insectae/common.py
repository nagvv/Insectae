from bisect import bisect
from inspect import Parameter, signature
from typing import Any, Callable, Dict, List, TypeVar

import numpy as np
from numpy.typing import NDArray

from .typing import Environment, Evaluable

_T = TypeVar("_T")


def evalf(param: Evaluable[_T], time: int, rng: np.random.Generator) -> _T:
    if callable(param):
        return param(time, rng)
    return param


# Evaluables (i.e. used with evalf)
class ExpCool:
    def __init__(self, x0: float, q: float) -> None:
        self.x = x0
        self.q = q
        self.gen = 0

    def __call__(self, time: int, rng: np.random.Generator) -> float:
        gen = time
        if gen > self.gen:
            self.gen = gen
            self.x *= self.q
        return self.x  # TODO: добавить возможность работы со списками или кортежами


class HypCool:
    def __init__(self, x0: float, deg: float) -> None:
        self.x0 = x0
        self.x = x0
        self.deg = deg
        self.gen = 0

    def __call__(self, time: int, rng: np.random.Generator) -> float:
        gen = time
        if gen > self.gen:
            self.gen = gen
            self.x = self.x0 / gen**self.deg
        return self.x


# Other common functions
def weighted_choice(values: List[Any], weights: List[float], rng: np.random.Generator):
    """
    Select a random element from `values` with probabilities proportional to `weights`.

    This function is the equivalent of `choices(values, weights=weights)[0]`
    where `choices` is from `random` module. Which is equivalent to
    `rng.choice(values, p=weights/sum(weights))`.

    We can't use `choices` because we want to use seeded numpy random number
    generator, but `choice` from numpy is extremely slow compared to `choices`
    from `random` module and this function.

    This function assumes that all input parameters are valid and does not validate them.

    Parameters
    ----------
    values : List[Any]
        The list of elements to choose from.
    weights : List[float]
        The list of weights corresponding to each element in `values`.
    rng : np.random.Generator
        A NumPy random generator instance for generating random numbers.
    """
    total = 0
    cum_weights = []
    for w in weights:
        total += w
        cum_weights.append(total)
    x = rng.random() * total
    i = bisect(cum_weights, x)
    return values[i]


def get_args_from_env(op: Callable, env: Environment) -> Dict[str, Any]:
    sig = signature(op)
    envargs = {}
    for idx, (key, par) in enumerate(sig.parameters.items()):
        if idx == 0:  # expected to be ind or pair of inds
            assert par.kind is Parameter.POSITIONAL_OR_KEYWORD
            continue
        if key == "key" or key == "twoway":  # provided explicitly or by pattern
            continue
        envargs[key] = env[key]
    return envargs


def samplex(n: int, m: int, exclude: List[int], rng: np.random.Generator) -> List[int]:
    s = list(set(range(n)) - set(exclude))
    return list(rng.choice(s, m, False))


def l2metrics(x: NDArray, y: NDArray) -> float:
    return float(np.linalg.norm(np.subtract(x, y)))
