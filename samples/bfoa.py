import insectae as ins
import numpy as np
from insectae.parallel import ThreadingExecutor


def target_func(x):
    return np.sum(np.square(x))


if __name__ == "__main__":
    goal = ins.ToMin()
    metrics = ins.Metrics(verbose=200)
    target = ins.RealTarget(
        metrics=metrics,
        target=target_func,
        dimension=10,
        bounds=(-10, 10),
    )
    stop = ins.StopMaxGeneration(1000, metrics=metrics)
    bfoa = ins.BacterialForagingAlgorithm(
        target=target,
        goal=goal,
        stop=stop,
        popSize=40,
        vel=ins.ExpCool(1.0, 0.99),
        gamma=0.1,
        probRotate=(0.01, 0.99),
        mu=0.01,
        opSelect=ins.TimedOp(ins.ProbOp(ins.ShuffledNeighbors(ins.Tournament(pwin=0.6)), 0.9), 10),
        opDisperse=ins.TimedOp(ins.ProbOp(ins.RandomRealVector(), 0.01), 20),
        opSignals=ins.TimedOp(ins.CalcSignals(shape=ins.ShapeClustering(0.00001)), 10),
        # only allNeighbors pattern (signals) will be parallelized
        # note, due to GIL it does not speedup effectively
        executor=ThreadingExecutor(processes=4, patterns={"allNeighbors"}),
    )
    # uncomment to disable signals between individuals
    # bfoa.opSignals = ins.NoSignals()

    timer = ins.Timer(metrics)
    ins.decorate(bfoa, [ins.TimeIt(timer)])

    bfoa.run()
    metrics.showTiming()
    metrics.show(log=True)
