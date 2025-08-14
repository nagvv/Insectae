from threading import Lock, Thread
from time import time
from typing import Any, Dict, List
from warnings import warn

from mpi4py import MPI

from ..alg_base import Algorithm
from ..typing import Environment, Individual
from .island_model import Communication


class MPICommunication(Communication):
    SEND_REQUESTS_KEY = "comm_send_requests"
    RECV_STATE_KEY = "comm_recv_worker_state"
    RECV_THREAD_KEY = "comm_recv_thread"
    TARGET_TO_RANK = "comm_target_to_rank"
    RANK_TO_TARGET = "comm_rank_to_target"
    COMM_TAG = "comm_tag"
    SERVICE_TAG = "comm_service_tag"

    def __init__(self, tag=0, service_tag=None, timeout=3) -> None:
        self.comm_tag = tag
        self.service_tag = service_tag or tag + 1
        self.timeout = timeout

    @staticmethod
    def send(data, targets: List[int], env: Environment) -> None:
        send_reqs = env.setdefault(MPICommunication.SEND_REQUESTS_KEY, [])
        # remove finished send requests
        finished_reqs, _ = MPI.Request.testsome(send_reqs)
        if finished_reqs is not None:
            for i in sorted(finished_reqs, reverse=True):
                del send_reqs[i]
        # start new send requests
        tgt_to_rank = env[MPICommunication.TARGET_TO_RANK]
        comm_tag = env[MPICommunication.COMM_TAG]
        for target in targets:
            send_reqs.append(
                MPI.COMM_WORLD.isend(data, dest=tgt_to_rank[target], tag=comm_tag)
            )

    @staticmethod
    def recv(env: Environment) -> List[Any]:
        queue = env[MPICommunication.RECV_STATE_KEY]["queue"]
        received = []
        while len(queue) > 0:
            received.append(queue.pop(0))
        return received

    def decorate(self, alg: Algorithm) -> None:
        if "MPICommunication" in alg.decorators:
            return

        alg.env[MPICommunication.COMM_TAG] = self.comm_tag
        alg.env[MPICommunication.SERVICE_TAG] = self.service_tag
        alg.addProcedure("start", self._init)
        alg.addProcedure("finish", self._deinit)
        alg.decorators.append("MPICommunication")

    def _init(self, _: List[Individual], env: Environment) -> None:
        # find out which island on which rank sits; we can not use collective
        # operations here, because it is not guaranteed that the amount of
        # islands is equal to the amount of MPI workers
        my_rank = MPI.COMM_WORLD.Get_rank()
        comm_size = MPI.COMM_WORLD.Get_size()
        islands_count = env["im_topo_size"]
        if islands_count > comm_size:
            # there are more islands than workers
            raise RuntimeError(
                f"there are more islands than MPI processes: {islands_count} vs {comm_size}"
            )
        if comm_size > islands_count:
            warn(
                f"there are more MPI processes than islands ({comm_size} vs {islands_count}) "
                f"it is highly recommended that these numbers match, otherwise "
                f"the operation may be unstable;"
            )
        test_reqs = []
        for tgt_rank in range(comm_size):
            if tgt_rank == my_rank:
                continue
            test_reqs.append(
                MPI.COMM_WORLD.isend(
                    (env["im_topo_idx"], my_rank), dest=tgt_rank, tag=self.service_tag
                )
            )
        target_to_rank = {env["im_topo_idx"]: my_rank}
        rank_to_target = {my_rank: env["im_topo_idx"]}
        recv_reqs = [
            MPI.COMM_WORLD.irecv(tag=self.service_tag) for _ in range(islands_count - 1)
        ]
        waiting_start_tp = time()
        while True:  # active waiting
            if len(recv_reqs) == 0:
                break
            if time() - waiting_start_tp > self.timeout:
                raise RuntimeError("could not establish connection with neighbors")
            finished_reqs, datas = MPI.Request.testsome(recv_reqs)
            if datas is None:
                continue
            assert finished_reqs is not None
            for idx, rank in datas:
                target_to_rank[idx] = rank
                rank_to_target[rank] = idx
            for idx in sorted(finished_reqs, reverse=True):
                del recv_reqs[idx]
        MPI.Request.waitall(recv_reqs)
        MPI.Request.waitsome(test_reqs)
        env[MPICommunication.TARGET_TO_RANK] = target_to_rank
        env[MPICommunication.RANK_TO_TARGET] = rank_to_target
        # start pipe thread
        state = env[MPICommunication.RECV_STATE_KEY] = {
            "active": True,
            "cur_req": MPI.REQUEST_NULL,
            "queue": [],
            "lock": Lock(),
        }
        recv_t = env[MPICommunication.RECV_THREAD_KEY] = Thread(
            target=self.recv_worker, args=[state, rank_to_target]
        )
        recv_t.start()

    @staticmethod
    def _deinit(_: List[Individual], env: Environment) -> None:
        send_reqs = env.setdefault(MPICommunication.SEND_REQUESTS_KEY, [])
        MPI.Request.waitall(send_reqs)  # wait till all sends are exhausted
        # improvised barrier to ensure sends are exhausted on every island
        barrier_recv_reqs = [
            MPI.COMM_WORLD.irecv(tag=env[MPICommunication.SERVICE_TAG])
            for _ in range(env["im_topo_size"] - 1)
        ]
        for idx in range(env["im_topo_size"]):
            if idx == env["im_topo_idx"]:
                continue
            MPI.COMM_WORLD.send(
                None,
                env[MPICommunication.TARGET_TO_RANK][idx],
                env[MPICommunication.SERVICE_TAG],
            )
        MPI.Request.waitall(barrier_recv_reqs)
        state = env[MPICommunication.RECV_STATE_KEY]
        with state["lock"]:
            state["active"] = False
            try:
                if state["cur_req"] != MPI.REQUEST_NULL:
                    state["cur_req"].cancel()
            except MPI.Exception:
                pass
        env[MPICommunication.RECV_THREAD_KEY].join()
        del env[MPICommunication.RECV_THREAD_KEY]  # unpickleable
        del env[MPICommunication.RECV_STATE_KEY]  # unpickleable
        # there can be some active send requests, but we can do nothing about
        # them since cancelling send request is not allowed anymore
        del env[MPICommunication.SEND_REQUESTS_KEY]  # unpickleable

    def recv_worker(self, state: dict, rank_to_target: Dict[int, int]) -> None:
        while True:
            with state["lock"]:
                if not state["active"]:
                    break
                recv_req = state["cur_req"] = MPI.COMM_WORLD.irecv(tag=self.comm_tag)
            status = MPI.Status()
            value = recv_req.wait(status)
            if not status.cancelled:
                state["queue"].append((rank_to_target[status.Get_source()], value))
