import insectae as ins
import numpy as np


def target_func(x):
    return np.sum(np.square(x))


if __name__ == "__main__":
    goal = ins.ToMin()
    metrics = ins.Metrics(goal=goal, verbose=200)
    target = ins.RealTarget(
        metrics=metrics, target=target_func, dimension=10, bounds=(-10, 10)
    )
    stop = ins.StopMaxGeneration(1000, metrics=metrics)
    gsa = ins.GravitationalSearchAlgorithm(
        g_init=5,
        g_decay_stop_it=1000,
        alpha=20,
        delta=0.01,
        target=target,
        popSize=40,
        goal=goal,
        stop=stop,
    )
    tm = ins.Timer(metrics=metrics)
    ins.decorate(gsa, [ins.TimeIt(tm), ins.AddElite(3)])
    gsa.run()
    metrics.showTiming()
    metrics.show(log=True)
