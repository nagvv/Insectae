from itertools import starmap
from typing import Any, Callable, Iterable, Tuple, Set, Optional

from .patterns import evaluate, foreach, neighbors, pairs, pop2ind, reducePop, signals

# FIXME/NOTE/TODO custom pattern implementations should be decorated with timing?
class BaseExecutor:
    def __init__(self, patterns: Optional[Set[str]] = None, *args, **kwargs) -> None:
        self.patterns = patterns if patterns is not None else {"evaluate"}
        super().__init__(*args, **kwargs)

    def starmap(
        self, fn: Callable[..., Any], fnargs: Iterable[Tuple], **kwargs
    ) -> Iterable[Any]:
        return starmap(fn, fnargs)

    def evaluate(self, *args, **kwargs):
        return evaluate(
            *args,
            executor=self if "evaluate" in self.patterns else None,
            **kwargs
        )

    def foreach(self, *args, **kwargs):
        return foreach(
            *args,
            executor=self if "foreach" in self.patterns else None,
            **kwargs
        )

    def neighbors(self, *args, **kwargs):
        return neighbors(
            *args,
            executor=self if "neighbors" in self.patterns else None,
            **kwargs
        )

    def pairs(self, *args, **kwargs):
        return pairs(
            *args,
            executor=self if "pairs" in self.patterns else None,
            **kwargs
        )

    def pop2ind(self, *args, **kwargs):
        return pop2ind(
            *args,
            executor=self if "pop2ind" in self.patterns else None,
            **kwargs
        )

    def reducePop(self, *args, **kwargs):
        return reducePop(
            *args,
            executor=self if "reducePop" in self.patterns else None,
            **kwargs
        )

    def signals(self, *args, **kwargs):
        return signals(
            *args,
            executor=self if "signals" in self.patterns else None,
            **kwargs
        )
