import insectae as ins
import numpy as np
from insectae.island_model import IslandModel, MPICommunication
from mpi4py.futures import MPIPoolExecutor


def one_min(x):
    return np.sum(x)


if __name__ == "__main__":
    algs = []
    goal = ins.ToMin()
    for i in range(4):
        metrics = ins.Metrics(verbose=100, goal=goal)
        target = ins.BinaryTarget(metrics=metrics, target=one_min, dimension=500)
        stop = ins.StopMaxGeneration(500, metrics=metrics)
        ga = ins.GeneticAlgorithm(
            opSelect=ins.ProbOp(ins.ShuffledNeighbors(ins.Tournament(pwin=0.9)), 0.5),
            opCrossover=ins.ShuffledNeighbors(ins.UniformCrossover(pswap=0.3)),
            opMutate=ins.BinaryMutation(prob=0.001),
            target=target,
            stop=stop,
            popSize=60,
            goal=goal,
            rng_seed=42 + i,
        )
        tm = ins.Timer(metrics=metrics)
        ins.decorate(ga, [ins.TimeIt(tm)])
        algs.append(ga)

    # process spawning may not work on some mpi implementations (ex: ms-mpi)
    # in that case run the sample as follows:
    # mpiexec -n 5 python3 -m mpi4py.futures ./island_model.pso.py
    with MPIPoolExecutor(max_workers=4) as executor:
        im = IslandModel(*algs, communication=MPICommunication(), executor=executor)
        # if only_workers=True then the main process won't be used to execute
        # an island; it is needed when island model is used with mpi process
        # spawning, since then spawned processes will be having a different
        # MPI_COMM_WORLD communicator that does not include the main process
        islands = im.run(only_workers=True)
        for island in islands:
            island.target.metrics.showTiming()
