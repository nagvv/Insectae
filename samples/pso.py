import insectae as ins
import numpy as np
from insectae.parallel import MultiprocessingExecutor, ThreadingExecutor


# Note: target function must be pickleable to allow parallelization through
# executors, except for ThreadingExecutor, which can work with non-pickleable
# functions just fine
def target_func(x):
    return np.sum(np.square(x))


if __name__ == "__main__":
    goal = ins.ToMin()
    metrics = ins.Metrics(goal=goal, verbose=200)
    target = ins.RealTarget(
        metrics=metrics, target=target_func, dimension=10, bounds=[-10, 10]
    )
    stop = ins.StopMaxGeneration(1000, metrics=metrics)
    executor = None
    # executor = ThreadingExecutor(processes=4)
    # executor = MultiprocessingExecutor(processes=4, chunksize=10)
    pso = ins.ParticleSwarmOptimization(
        target=target,
        goal=goal,
        stop=stop,
        popSize=40,
        alphabeta=ins.LinkedAlphaBeta(0.1),
        gamma=0.95,
        delta=0.01,
        executor=executor,
    )

    # pso.opLimitVel = ins.MaxAmplitude(0.95)
    pso.opLimitVel = ins.MaxAmplitude(ins.ExpCool(0.5, 0.999))

    tm = ins.Timer(metrics=metrics)
    ins.decorate(pso, [ins.TimeIt(tm), ins.AddElite(2)])
    pso.run()

    metrics.showTiming()
    metrics.show(log=True)
