import numpy as np
import insectae as ins

goal = ins.ToMin()
metrics = ins.Metrics(goal=goal, verbose=200)
target = ins.RealTarget(
    metrics=metrics,
    target=lambda x: np.sum(np.square(x)),
    dimension=10,
    bounds=[-10, 10]
)
stop = ins.StopMaxGeneration(1000, metrics=metrics)

pso = ins.ParticleSwarmOptimization(
    target=target,
    goal=goal,
    stop=stop,
    popSize=40,
    alphabeta=ins.LinkedAlphaBeta(0.1),
    gamma=0.95,
    delta=0.01
)

#pso.opLimitVel = im.MaxAmplitude(0.95)
#pso.opLimitVel = ins.MaxAmplitude(ins.expCool(0.5, 0.999))

tm = ins.Timer(metrics=metrics)

# ins.decorate(pso, [ins.TimeIt(tm)])
ins.decorate(pso, [ins.TimeIt(tm), ins.AddElite(2)])
pso.run()

metrics.showTiming()
metrics.show(log=True)
