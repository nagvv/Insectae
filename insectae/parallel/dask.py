from typing import Any, Callable, Iterable, Tuple

from dask.distributed import Client as DaskClient

from ..executor import BaseExecutor


class DaskExecutor(BaseExecutor, DaskClient):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def starmap(
        self, fn: Callable[..., Any], fnargs: Iterable[Tuple], **kwargs
    ) -> Iterable[Any]:
        # fnargs is a iterable of tuples, where one tuple holds arguments
        # for a one job, but dask client expects multiple iterables, where
        # i-th iterable holds i-th arguments of the every job
        futures = DaskClient.map(self, fn, *zip(*fnargs), **kwargs)
        return (fut.result() for fut in futures)
