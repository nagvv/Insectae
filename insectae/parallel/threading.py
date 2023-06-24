from multiprocessing.pool import ThreadPool
from typing import Any, Callable, Iterable, Tuple

from ..executor import BaseExecutor


class ThreadingExecutor(BaseExecutor, ThreadPool):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def starmap(
        self, fn: Callable[..., Any], fnargs: Iterable[Tuple], **kwargs
    ) -> Iterable[Any]:
        return ThreadPool.starmap(self, fn, fnargs, **kwargs)
