import numpy as np

import insectae as ins

goal = ins.ToMin()
metrics = ins.Metrics(goal=goal, verbose=200)
target = ins.RealTarget(
    metrics=metrics,
    target=lambda x: np.sum(np.square(x)),
    dimension=10,
    bounds=[-10, 10],
)
stop = ins.StopMaxGeneration(1000, metrics=metrics)

de = ins.DifferentialEvolution(
    opMakeProbe=ins.ProbeClassic(0.8),
    opCrossover=ins.UniformCrossover(0.9),
    opSelect=ins.Tournament(1.0),
    target=target,
    goal=goal,
    stop=stop,
    popSize=40,
)

# de.opMakeProbe = ins.ProbeBest(0.8)
# de.opMakeProbe = ins.ProbeCur2Best(0.8)
# de.opMakeProbe = ins.ProbeBest2(0.8)
# de.opMakeProbe = ins.probeRandom5(0.8)

ins.decorate(de, [ins.TimeIt(ins.Timer(metrics))])

de.run()
metrics.showTiming()
metrics.show(log=True)
