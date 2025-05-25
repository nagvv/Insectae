import insectae as ins
import numpy as np


def one_min(x):
    return np.sum(x)


if __name__ == "__main__":
    goal = ins.ToMin()
    metrics = ins.Metrics(verbose=100, goal=goal)
    target = ins.BinaryTarget(metrics=metrics, target=one_min, dimension=500)
    stop = ins.StopMaxGeneration(500, metrics=metrics)
    umda = ins.UnivariateMarginalDistributionAlgorithm(
        # opSelect=ins.ShuffledNeighbors(ins.Tournament(pwin=1.0)),
        opSelect=ins.Sorted(ins.SelectLeft(0.7)),
        target=target,
        stop=stop,
        popSize=60,
        goal=goal,
    )
    tm = ins.Timer(metrics=metrics)
    ins.decorate(umda, [ins.TimeIt(tm)])
    umda.run()
    metrics.showTiming()
    metrics.show(bottom=0)
