import insectae as ins
import numpy as np


def target_func(x):
    return np.sum(x)


if __name__ == "__main__":
    goal = ins.ToMax()
    metrics = ins.Metrics(goal="max", verbose=200)
    target = ins.BinaryTarget(metrics=metrics, target=target_func, dimension=100)
    stop = ins.StopMaxGeneration(1000, metrics=metrics)
    opLocal = ins.BinaryMutation(ins.ExpCool(0.1, 0.99))
    opGlobal = ins.RandomBinaryVector()
    opProbs = ins.BinaryPlacesProbs(0.5, 0.9, pscout=0.1)

    bees = ins.BeesAlgorithm(
        beesNum=20,
        opLocal=opLocal,
        opGlobal=opGlobal,
        opProbs=opProbs,
        target=target,
        goal=goal,
        stop=stop,
        popSize=20,
    )
    ins.decorate(bees, [ins.TimeIt(ins.Timer(metrics))])
    bees.run()
    metrics.showTiming()
    metrics.show()
