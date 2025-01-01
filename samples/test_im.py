import numpy as np
from mpi4py.futures import MPIPoolExecutor

import insectae as ins
from insectae.island_model import *


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
            opSelect=ins.Shuffled(ins.ProbOp(ins.Tournament(pwin=0.9), 0.5)),
            opCrossover=ins.Shuffled(ins.UniformCrossover(pswap=0.3)),
            opMutate=ins.BinaryMutation(prob=0.001),
            target=target,
            stop=stop,
            popSize=60,
            goal=goal,
            rng_seed=42 + i
        )
        tm = ins.Timer(metrics=metrics)
        ins.decorate(ga, [ins.TimeIt(tm)])
        algs.append(ga)

    with MPIPoolExecutor() as executor:
        im = IslandModel(*algs, communication=MPICommunication(), executor=executor)
        islands = im.run()
        for island in islands:
            island.target.metrics.showTiming()
