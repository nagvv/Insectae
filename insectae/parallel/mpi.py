from typing import Any, Callable, Iterable, Tuple

from mpi4py.futures import MPIPoolExecutor

from ..executor import BaseExecutor


class MPIExecutor(BaseExecutor, MPIPoolExecutor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def starmap(
        self, fn: Callable[..., Any], fnargs: Iterable[Tuple], **kwargs
    ) -> Iterable[Any]:
        return MPIPoolExecutor.starmap(self, fn, fnargs, **kwargs)
