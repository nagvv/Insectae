import numpy as np

import insectae as ins

goal = ins.ToMin()
metrics = ins.Metrics(verbose=100, goal=goal)
target = ins.BinaryTarget(metrics=metrics, target=lambda x: np.sum(x), dimension=500)
stop = ins.StopMaxGeneration(1000, metrics=metrics)

ga = ins.UnivariateMarginalDistributionAlgorithm(
    # opSelect=ins.Shuffled(ins.Tournament(pwin=1.0)),
    opSelect=ins.Sorted(ins.SelectLeft(0.7)),
    target=target,
    stop=stop,
    popSize=60,
    goal=goal
)

tm = ins.Timer(metrics=metrics)
ins.decorate(ga, [ins.TimeIt(tm)])

ga.run()

metrics.showTiming()
metrics.show(bottom=0)
