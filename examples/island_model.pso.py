import insectae as ins
import numpy as np
from insectae.island_model import IslandModel, MPICommunication
from mpi4py.futures import MPIPoolExecutor


def square_sum(x):
    return np.sum(np.square(x))


if __name__ == "__main__":
    algs = []
    goal = ins.ToMin()
    for i in range(4):
        metrics = ins.Metrics(verbose=100, goal=goal)
        target = ins.RealTarget(
            metrics=metrics, target=square_sum, dimension=10, bounds=[-10, 10]
        )
        stop = ins.StopMaxGeneration(1000, metrics=metrics)
        pso = ins.ParticleSwarmOptimization(
            target=target,
            goal=goal,
            stop=stop,
            popSize=40,
            alphabeta=ins.LinkedAlphaBeta(0.1),
            gamma=0.95,
            delta=0.01,
            rng_seed=42 + i,
        )
        tm = ins.Timer(metrics=metrics)
        ins.decorate(pso, [ins.TimeIt(tm)])
        algs.append(pso)

    # process spawning may not work on some mpi implementations (ex: ms-mpi)
    # in that case run the sample as follows:
    # mpiexec -n 5 python3 -m mpi4py.futures ./island_model.pso.py
    with MPIPoolExecutor(max_workers=4) as executor:
        im = IslandModel(
            *algs,
            communication=MPICommunication(),
            executor=executor,
        )
        # if only_workers=True then the main process won't be used to execute
        # an island; it is needed when island model is used with mpi process
        # spawning, since then spawned processes will be having a different
        # MPI_COMM_WORLD communicator that does not include the main process
        islands = im.run(only_workers=True)
        for island in islands:
            island.target.metrics.showTiming()
