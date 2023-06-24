from multiprocessing.pool import Pool
from typing import Any, Callable, Iterable, Tuple

from ..executor import BaseExecutor


class MultiprocessingExecutor(BaseExecutor, Pool):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def starmap(
        self, fn: Callable[..., Any], fnargs: Iterable[Tuple], **kwargs
    ) -> Iterable[Any]:
        return Pool.starmap(self, fn, fnargs, **kwargs)
