import numpy as np

import insectae as ins

metrics = ins.Metrics(verbose=200)
target = ins.RealTarget(
    metrics=metrics,
    target=lambda x: np.sum(np.square(x)),
    dimension=10,
    bounds=[-10, 10],
)
stop = ins.StopMaxGeneration(1000, metrics=metrics)

cso = ins.CompetitiveSwarmOptimizer(
    socialFactor=0.1,
    delta=0.01,
    target=target,
    stop=stop,
    popSize=40,
)

timer = ins.Timer(metrics)
ins.decorate(cso, [ins.TimeIt(timer)])

cso.run()

metrics.showTiming()
metrics.show(log=True)
