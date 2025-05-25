import insectae as ins
import numpy as np


def target_func(x):
    return np.sum(x)


if __name__ == "__main__":
    metrics = ins.Metrics(goal="max", verbose=100)
    target = ins.BinaryTarget(metrics=metrics, target=target_func, dimension=100)
    # stop = ins.StopMaxGeneration(1000, metrics=metrics)
    stop = ins.StopValue(98, 1000, metrics=metrics)
    sa = ins.SimulatedAnnealing(
        theta=ins.ExpCool(1.0, 0.99),
        opMove=ins.BinaryMutation(prob=0.01),
        target=target,
        goal="max",
        stop=stop,
        popSize=20,
    )
    tm = ins.Timer(metrics)
    ins.decorate(sa, [ins.TimeIt(tm)])
    sa.run()
    metrics.showTiming()
    metrics.show(log=False)
