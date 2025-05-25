import insectae as ins
import numpy as np


def target_func(x):
    return np.sum(np.square(x))


if __name__ == "__main__":
    goal = ins.ToMin()
    metrics = ins.Metrics(goal=goal, verbose=200)
    target = ins.RealTarget(
        metrics=metrics,
        target=target_func,
        dimension=10,
        bounds=[-10, 10],
    )
    stop = ins.StopMaxGeneration(1000, metrics=metrics)
    ffa = ins.FireflyAlgorithm(
        alpha=0.01,
        alphabest=0.002,
        betamin=1,
        gamma=0.01,
        theta=1.0,
        target=target,
        goal=goal,
        stop=stop,
        popSize=40,
    )
    tm = ins.Timer(metrics)
    ins.decorate(ffa, [ins.TimeIt(tm), ins.AddElite(2)])
    ffa.run()
    metrics.showTiming()
    metrics.show(log=True)
