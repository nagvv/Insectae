import insectae as ins
import numpy as np
from insectae.parallel import MultiprocessingExecutor


def one_min(x):
    return np.sum(x)


if __name__ == "__main__":
    goal = ins.ToMin()
    metrics = ins.Metrics(verbose=100, goal=goal)
    target = ins.BinaryTarget(metrics=metrics, target=one_min, dimension=500)
    stop = ins.StopMaxGeneration(500, metrics=metrics)
    ga = ins.GeneticAlgorithm(
        opSelect=ins.ProbOp(ins.ShuffledNeighbors(ins.Tournament(pwin=0.9)), 0.5),
        opCrossover=ins.ShuffledNeighbors(ins.UniformCrossover(pswap=0.3)),
        opMutate=ins.BinaryMutation(prob=0.001),
        target=target,
        stop=stop,
        popSize=240,
        goal=goal,
        rng_seed=42,
        # only foreach pattern (mutate) will be parallelized
        executor=MultiprocessingExecutor(
            processes=4, chunksize=60, patterns={"foreach"}
        ),
    )
    x1 = ins.UniformCrossover(pswap=0.3)
    x2 = ins.SinglePointCrossover()
    x3 = ins.DoublePointCrossover()
    # ga.opCrossover = ins.Mixture(list(map(ins.Selected, [x1, x2, x3])), [0.2, 0.2, 0.2])
    ga.opCrossover = ins.Mixture(
        list(map(ins.ShuffledNeighbors, [x1, x2, x3])), [0.2, 0.2, 0.2]
    )
    tm = ins.Timer(metrics=metrics)
    ins.decorate(ga, [ins.TimeIt(tm)])
    ga.run()
    metrics.showTiming()
    metrics.show(bottom=0)
