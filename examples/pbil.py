import insectae as ins
import numpy as np


def one_min(x):
    return np.sum(x)


if __name__ == "__main__":
    goal = ins.ToMin()
    metrics = ins.Metrics(verbose=100, goal=goal)
    target = ins.BinaryTarget(metrics=metrics, target=one_min, dimension=500)
    stop = ins.StopMaxGeneration(500, metrics=metrics)
    pbil = ins.PopulationBasedIncrementalLearning(
        probMutate=ins.RealMutation(delta=0.01),
        n_best=5,
        n_worst=3,
        p_max=0.95,
        p_min=0.01,
        learning_rate=0.1,
        target=target,
        stop=stop,
        popSize=60,
        goal=goal,
    )
    tm = ins.Timer(metrics=metrics)
    ins.decorate(pbil, [ins.TimeIt(tm)])
    pbil.run()
    metrics.showTiming()
    metrics.show(bottom=0)
