import numpy as np

import insectae as ins

metrics = ins.Metrics(goal="max", verbose=100)

target = ins.BinaryTarget(metrics=metrics, target=lambda x: np.sum(x), dimension=100)
# target = ins.RealTarget(metrics=metrics, target=lambda x: np.sum(np.square(x)), dimension=10, bounds=[-10, 10])
stop = ins.StopMaxGeneration(500, metrics=metrics)
# stop = ins.StopValue(98, 1000, metrics=metrics)

sa = ins.SimulatedAnnealing(
    theta=ins.ExpCool(1.0, 0.99),
    opMove=ins.BinaryMutation(prob=0.01),
    target=target,
    goal="max",
    stop=stop,
    popSize=20,
)

# sa.opMove = ins.RealMutation(delta=ins.HypCool(0.1, 0.07))

tm = ins.Timer(metrics)

ins.decorate(sa, [ins.TimeIt(tm)])

sa.run()
metrics.showTiming()
metrics.show(log=True)
