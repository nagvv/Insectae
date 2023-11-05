import numpy as np

import insectae as ins

metrics = ins.Metrics(verbose=100)
# target = ins.BinaryTarget(metrics=metrics, target=lambda x: np.sum(x), dimension=100)
target = ins.BinaryTarget(metrics=metrics, target=lambda x: np.sum(x), dimension=500)
# target = ins.RealTarget(metrics=metrics, target=lambda x: np.sum(np.square(x)), dimension=10, bounds=[-10, 10])
stop = ins.StopMaxGeneration(1000, metrics=metrics)

ga = ins.GeneticAlgorithm(
    opSelect=ins.Shuffled(ins.ProbOp(ins.Tournament(pwin=0.9), 0.5)),
    opCrossover=ins.Shuffled(ins.UniformCrossover(pswap=0.3)),
    opMutate=ins.BinaryMutation(prob=0.001),
    target=target,
    stop=stop,
    popSize=60,
)

# ga.opSelect = ins.Selected(ins.ProbOp(ins.Tournament(pwin=0.9), 0.5))
# x1 = ins.UniformCrossover(pswap=0.3)
# x2 = ins.SinglePointCrossover()
# x3 = ins.DoublePointCrossover()
# x = ins.Mixture([x1, x2, x3], [0.2, 0.2, 0.2])
# ga.opCrossover = ins.Selected(x2)
# ga.opCrossover = ins.Shuffled(ins.TimedOp(x1, 10))
# ga.opCrossover = ins.Shuffled(ins.ProbOp(x1, 0.1))
# ga.opCrossover = ins.Shuffled(x)


class BinaryRanks:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, **xt):
        r = xt["inds"][0]["_rank"]
        ps = xt["popSize"]
        if r / ps < 0.5:
            return self.a
        return self.b
        # return self.a + (self.b - self.a) * (r + 1) / ps


class ExpRanks:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, inds, env):
        r = inds[0]["_rank"]
        ps = env["popSize"]
        return self.a * (self.b / self.a) ** (r / (ps - 1))


# ga.opMutate = ins.RealMutation(delta=ExpRanks(0.0000000000001, 1))

# ga.opMutate = ins.RealMutation(delta=ins.ExpCool(0.5, 0.99))
# ga.opMutate = ins.RealMutation(delta=ins.HypCool(0.1, 0.25))
# ga.opMutate = ins.BinaryMutation(prob=ins.ExpCool(0.1, 0.99))

ins.decorate(ga, [ins.RankIt()])

# ins.decorate(ga, [ins.AddElite(5), ins.TimeIt(ins.Timer(metrics)), ins.RankIt()])
# ins.decorate(ga, ins.AddElite(0.25))

# tm = ins.Timer(metrics=metrics)
# ins.decorate(ga, [ins.TimeIt(tm)])

ga.run()

metrics.showTiming()
metrics.show(bottom=0)
