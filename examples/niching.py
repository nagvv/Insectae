import insectae as ins
import numpy as np


def u(val: int, x) -> float:
    if val == 0 or val == 6:
        return 1.0
    elif val == 1 or val == 5:
        return 0.0
    elif val == 2 or val == 4:
        return 0.640576 / 2.0
    elif val == 3:
        return 0.640576
    else:
        raise RuntimeError("bad unitation u val", val, x)


def M7(x):  # 32 global maximax
    res = 0
    for i in range(5):
        inner_sum = 0
        for j in range(6):
            inner_sum += x[6 * i + j]
        res += u(inner_sum, x)
    return res


def dist_func(a, b):  # normalized Hamming
    return np.count_nonzero(a != b) / 30.0


if __name__ == "__main__":
    goal = ins.ToMax()
    metrics = ins.Metrics(goal=goal, verbose=20)
    target = ins.BinaryTarget(
        metrics=metrics,
        target=M7,
        dimension=30,
    )
    stop = ins.StopMaxGeneration(100, metrics=metrics)
    ga = ins.GeneticAlgorithm(
        opSelect=ins.Sorted(ins.SelectLeft(0.5)),
        opCrossover=ins.ShuffledNeighbors(ins.SinglePointCrossover()),
        opMutate=ins.BinaryMutation(prob=0.002),
        target=target,
        stop=stop,
        popSize=600,
        goal=goal,
    )
    tm = ins.Timer(metrics=metrics)
    sharing1 = ins.AddFitnessSharing(sigma=0.2, alpha=1.0, dist_func=dist_func)
    sharing2 = ins.AddClearing(sigma=0.2, capacity=1, dist_func=dist_func)

    # ins.decorate(ga, [ins.TimeIt(tm), ins.AddElite(100)])
    # ins.decorate(ga, [ins.TimeIt(tm), sharing1, ins.AddElite(100)])
    ins.decorate(ga, [ins.TimeIt(tm), sharing2, ins.AddElite(100)])

    ga.run()
    metrics.showTiming()

    np.set_printoptions(precision=3, floatmode="fixed", suppress=True)
    arr = []
    maximas_found = set()
    locals_found = set()
    for ind in ga.population:
        arr.append(ind["x"])
        if ind["f"] == 5:
            maximas_found.add(tuple(ind["x"]))
        if ind["f"] > 3.2:
            locals_found.add(tuple(ind["x"]))
    arr = np.array(arr)
    print("uniques", len(np.unique(arr.round(decimals=1), axis=0)))
    print("maximas found", len(maximas_found))
    print("locals found", len(locals_found))
