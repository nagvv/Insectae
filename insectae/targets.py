from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from .typing import Individual, Environment
from .metrics import Metrics


class Target:
    def __init__(
        self,
        encoding: str,
        target: Callable[[Any], Any],
        dimension: int,
        metrics: Metrics,
    ) -> None:
        self.encoding = encoding
        self.target = target
        self.dimension = dimension
        self.metrics = metrics

    def __call__(self, x: Any, f: Any, reEval: bool) -> Any:
        if reEval:
            f = self.target(x)
        self.metrics.newEval(x, f, reEval)
        return f

    def get_func(self) -> Callable[[Any], Any]:
        return self.target

    def update(self, x: Any, f: Any, reEval: bool) -> None:
        self.metrics.newEval(x, f, reEval)

    def defaultInit(self) -> Callable[..., None]:
        raise NotImplementedError


class BinaryTarget(Target):
    def __init__(
        self,
        target: Callable[[Any], Any],
        dimension: int,
        metrics: Metrics,
    ) -> None:
        super().__init__("binary", target, dimension, metrics)

    def defaultInit(self) -> Callable[..., None]:
        return RandomBinaryVector()


# TODO: consider adding base class for these random generators
class RandomBinaryVector:
    def __call__(
        self,
        ind: Individual,
        target: Target,
        key: str,
        env: Optional[Environment] = None
    ) -> None:
        dim = target.dimension
        ind[key] = np.random.randint(2, size=dim)


class RealTarget(Target):
    def __init__(
        self,
        target: Callable[[Any], Any],
        dimension: int,
        metrics: Metrics,
        bounds: Tuple[float, float],
    ) -> None:
        super().__init__("real", target, dimension, metrics)
        self.bounds = bounds

    def defaultInit(self) -> Callable[..., None]:
        return RandomRealVector()


class RandomRealVector:
    def __init__(
        self, bounds: Optional[Union[float, Tuple[float, float]]] = None
    ) -> None:
        self.bounds: Optional[Tuple[float, float]] = None
        if isinstance(bounds, float):
            self.bounds = (-bounds, bounds)
        else:
            self.bounds = bounds  # either tuple or None

    def __call__(
        self,
        ind: Individual,
        target: RealTarget,
        key: str,
        env: Optional[Environment] = None
    ) -> None:
        dim = target.dimension
        if self.bounds is not None:
            low, high = self.bounds
        else:
            low, high = target.bounds
        ind[key] = low + (high - low) * np.random.rand(dim)


class PermutationTarget(Target):
    def __init__(
        self,
        target: Callable[[Any], Any],
        dimension: int,
        metrics: Metrics,
    ) -> None:
        super().__init__("permutation", target, dimension, metrics)

    def defaultInit(self) -> Callable[..., None]:
        return RandomPermutation()


class RandomPermutation:
    def __call__(
        self,
        ind: Individual,
        target: Target,
        key: str,
        env: Optional[Environment] = None
    ) -> None:
        dim = target.dimension
        ind[key] = np.array(list(range(dim)))
        np.random.shuffle(ind[key])
