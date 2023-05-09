import numpy as np

import insectae as ins

goal = ins.ToMin()
metrics = ins.Metrics(verbose=200)
target = ins.RealTarget(
    metrics=metrics,
    target=lambda x: np.sum(np.square(x)),
    dimension=10,
    bounds=[-10, 10],
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
    opSelect=ins.Shuffled(ins.TimedOp(ins.ProbOp(ins.Tournament(pwin=0.6), 0.9), 10)),
    opDisperse=ins.TimedOp(ins.ProbOp(ins.RandomRealVector(), 0.01), 20),
    opSignals=ins.TimedOp(ins.CalcSignals(shape=ins.ShapeClustering(0.00001)), 10),
)

# bfoa.opDisperse = ins.ProbOp(ins.RandomRealVector(), 0.0001)

# bfoa.opSignals = ins.NoSignals()
# bfoa.opSignals = ins.TimedOp(ins.CalcSignals(shape=ins.ShapeClustering(ins.ExpCool(0.001, 0.999), "min")), 10)

timer = ins.Timer(metrics)
# ins.decorate(bfoa, [ins.TimeIt(timer), ins.AddElite(1)])
ins.decorate(bfoa, [ins.TimeIt(timer)])

bfoa.run()
metrics.showTiming()
metrics.show(log=True)
