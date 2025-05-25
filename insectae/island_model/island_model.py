from abc import ABC, abstractmethod
from operator import itemgetter
from typing import Any, Dict, List, Union

from ..alg_base import Algorithm
from ..decorators import decorate
from ..typing import Environment, Individual


class Communication(ABC):
    @staticmethod
    @abstractmethod
    def send(data: Any, targets: List[int], env: Environment) -> None:
        pass

    @staticmethod
    @abstractmethod
    def recv(env: Environment) -> List[Any]:
        pass

    @abstractmethod
    def decorate(self, alg: Algorithm) -> None:
        pass


class AddMigration:
    LAST_MIGRATED = "im_last_migrated"

    def __init__(
        self,
        comm: Communication,
        migration_size: int = 3,
        migration_interval: int = 10,
    ) -> None:
        self._comm = comm
        self._migration_size = migration_size
        self._migration_interval = migration_interval

    def __call__(self, alg: Algorithm) -> None:
        if "AddMigration" in alg.decorators:
            return

        alg.env[self.LAST_MIGRATED] = 0
        alg.addProcedure("exit", self._migrate)
        alg.decorators.append("AddMigration")

    def _migrate(self, population: List[Individual], env: Environment) -> None:
        env[self.LAST_MIGRATED] += 1
        if env[self.LAST_MIGRATED] < self._migration_interval:
            return

        keyf = env["solutionValueLabel"]
        # get the sorted indices of individuals from the best to worst
        sorted_idxs = sorted(
            range(len(population)),
            key=env["goal"].get_cmp_to_key(lambda v: population[v][keyf]),
        )
        selected = [population[idx] for idx in sorted_idxs[: self._migration_size]]
        self._comm.send(selected, env["im_topo_targets"], env)
        received = self._comm.recv(env)
        # use the last received data from each sender
        to_insert = []
        already_has = set()
        for sender, data in reversed(received):
            if sender in already_has:
                continue
            already_has.add(sender)
            to_insert.extend(data)
        # select the best from the received data and the current worsts
        to_insert.extend(
            [population[idx] for idx in sorted_idxs[-self._migration_size :]]
        )
        to_insert.sort(key=env["goal"].get_cmp_to_key(itemgetter(keyf)))
        for idx, ind in zip(
            sorted_idxs[-self._migration_size :], to_insert[: self._migration_size]
        ):
            population[idx] = ind
        env[self.LAST_MIGRATED] = 0


def make_island_model(
    *algorithms: Algorithm,
    communication: Communication,
    topology: Union[str, Dict[int, List[int]]] = "circle",
    migration_size: int = 3,
    migration_interval: int = 10,
) -> List[Algorithm]:
    count = len(algorithms)
    if count == 0:
        raise ValueError("no algorithm instance were provided")

    if isinstance(topology, str):
        if topology == "circle":
            topology = {i: [(i + 1) % count] for i in range(count)}
        else:
            raise ValueError(f"unknown topology '{topology}'")
    elif isinstance(topology, dict):
        if len(topology) != count:
            raise ValueError(
                f"topology size does not match islands count: "
                f"{len(topology)} != {count}"
            )
    else:
        raise TypeError(f"wrong topology type '{type(topology)}'")

    for idx, alg in enumerate(algorithms):
        alg.env["im_topo_idx"] = idx
        alg.env["im_topo_size"] = count
        alg.env["im_topo_targets"] = topology[idx]
        decorate(
            alg,
            [
                communication.decorate,
                AddMigration(
                    communication,
                    migration_size=migration_size,
                    migration_interval=migration_interval,
                ),
            ],
        )

    return list(algorithms)


class IslandModel:
    def __init__(
        self,
        *algorithms: Algorithm,
        communication: Communication,
        topology: Union[str, Dict[int, List[int]]] = "circle",
        executor: Any,
        **kwargs,
    ) -> None:
        self._executor = executor
        self._islands = make_island_model(
            *algorithms, communication=communication, topology=topology, **kwargs
        )

    def run(self, only_workers=False) -> List[Algorithm]:
        if only_workers:
            res = self._executor.map(self._worker, self._islands)
            self._islands = list(res)
        else:
            res = self._executor.map(self._worker, self._islands[1:])
            self._islands[0].run()
            self._islands[1:] = list(res)
        return self._islands

    @staticmethod
    def _worker(alg: Algorithm):
        alg.run()
        return alg
