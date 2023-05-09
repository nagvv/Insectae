import numpy as np
import insectae as ins

# goal = ins.ToMin()
# metrics = ins.Metrics(goal=goal, verbose=200)
# target = ins.RealTarget(metrics=metrics, target=lambda x: np.sum(np.square(x)), dimension=10, bounds=[-10, 10])
# stop = ins.StopMaxGeneration(1000, metrics=metrics)
# opLocal = ins.RealMutation(ins.ExpCool(1, 0.99))
# opGlobal = ins.RandomRealVector()
# opProbs = ins.UniformPlacesProbs(pscout=0.1)
# opProbs = ins.LinearPlacesProbs(0.9, pscout=0.1)

goal = ins.ToMax()
metrics = ins.Metrics(goal="max", verbose=200)
target = ins.BinaryTarget(metrics=metrics, target=lambda x: np.sum(x), dimension=100)
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
