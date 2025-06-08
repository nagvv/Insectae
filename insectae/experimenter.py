from copy import deepcopy
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .alg_base import Algorithm
from .alg_bees import BeesAlgorithm, UniformPlacesProbs
from .alg_bfoa import BacterialForagingAlgorithm, CalcSignals, ShapeClustering
from .alg_cso import CompetitiveSwarmOptimizer
from .alg_de import DifferentialEvolution, ProbeClassic
from .alg_eda import (PopulationBasedIncrementalLearning,
                      UnivariateMarginalDistributionAlgorithm)
from .alg_ffa import FireflyAlgorithm
from .alg_ga import GeneticAlgorithm
from .alg_gsa import GravitationalSearchAlgorithm
from .alg_pso import LinkedAlphaBeta, ParticleSwarmOptimization
from .alg_sa import SimulatedAnnealing
from .common import ExpCool
from .executor import BaseExecutor
from .goals import Goal
from .operators import (ProbOp, RealMutation, SelectLeft, ShuffledNeighbors,
                        Sorted, TimedOp, Tournament, UniformCrossover)
from .stops import StopMaxGeneration
from .targets import BinaryTarget, PermutationTarget, RealTarget, Target


class _wire_args_eval:
    def __init__(self, fn: Callable, **kwargs):
        self.fn = fn
        self.wires = kwargs

    def __call__(self, *args, **kwargs):
        return self.fn(
            *args,
            # FIXME: eval signature was changed in Python 3.13
            **{k: eval(v, globals(), kwargs) for k, v in self.wires.items()},
            **kwargs,
        )

class Experimenter:
    def __init__(
        self,
        target: Target,
        goal: Goal,
        pop_size: Union[int, Tuple[int]] = 20,
        executor: Optional[BaseExecutor] = None,
        runs_count: int = 10,
        iters_count: int = 100,
    ) -> None:
        self.target = target
        self.goal = goal
        self.metrics = target.metrics
        self.pop_size = pop_size
        self.executor = executor or BaseExecutor()
        self.executor.init({}, np.random.default_rng())  # TODO: make rng argument optional
        self.runs_count = runs_count
        self.iters_count = iters_count
        self.stop = StopMaxGeneration(self.iters_count, metrics=self.metrics)

    @staticmethod
    def build_and_run(builder: Callable[..., Algorithm]) -> Algorithm:
        alg = builder()
        alg.run()
        return alg

    def run(self):
        alg_builders = self._get_algorithm_builders(self.target)
        finished_algs = self.executor.starmap(self.build_and_run, zip(alg_builders))
        # fill results per run list
        results_per_run = []
        alg_names = set()
        for alg in finished_algs:
            metrics = alg.target.metrics
            sol_val_lbl = alg.env["solutionValueLabel"]
            results_per_run.append(
                {
                    "alg_name": type(alg).__name__,
                    "pop_size": alg.popSize,
                    "best_val": metrics.bestValue,
                    "best_solution": metrics.bestSolution,
                    "efs": metrics.efs,
                    "population": alg.population,
                    "avg_val": sum(ind[sol_val_lbl] for ind in alg.population)
                    / alg.popSize,
                    "vals_data": metrics.data
                }
            )
            alg_names.add(type(alg).__name__)

        # fill results per algorithm dictionary
        results_per_alg = {}
        for alg_name in alg_names:
            alg_runs = [r for r in results_per_run if r["alg_name"] == alg_name]
            alg_res = results_per_alg.setdefault(alg_name, {})
            alg_res["best_run_idx"] = results_per_run.index(
                min(alg_runs, key=self.goal.get_cmp_to_key(lambda r: r["best_val"]))
            )
            alg_res["avg_best_val"] = sum(r["best_val"] for r in alg_runs) / len(alg_runs)
            alg_res["avg_avg_val"] = sum(r["avg_val"] for r in alg_runs) / len(alg_runs)

        # fill total results
        result = {
            "best_run_idx": results_per_run.index(min(
                (results_per_run[r["best_run_idx"]] for r in results_per_alg.values()),
                key=self.goal.get_cmp_to_key(lambda r: r["best_val"])
            )),
            "best_avg_alg": min(
                results_per_alg.keys(),
                key=self.goal.get_cmp_to_key(lambda alg: results_per_alg[alg]["avg_best_val"])
            ),
            "avg_best_val": sum(r["avg_best_val"] for r in results_per_alg.values()) / len(results_per_alg),
            "avg_avg_val": sum(r["avg_avg_val"] for r in results_per_alg.values()) / len(results_per_alg),
        }

        return result, results_per_alg, results_per_run

    def _get_algorithm_builders(self, target: Target) -> List[Callable[..., Algorithm]]:
        _builders = []
        if isinstance(target, RealTarget):
            _builders = self._get_algorithm_builders_real()
        elif isinstance(target, BinaryTarget):
            _builders = self._get_algorithm_builders_binary()
        elif isinstance(target, PermutationTarget):
            _builders = self._get_algorithm_builders_permutation()
        else:
            raise ValueError("Unsupported target")
        pop_sizes = (
            self.pop_size if isinstance(self.pop_size, tuple) else (self.pop_size,)
        )
        alg_builders = []
        for pop_size in pop_sizes:
            common_args = {
                "target": self.target,
                "goal": self.goal,
                "stop": self.stop,
                "popSize": pop_size,
                "opInit": None,  # default, i.e. target.defaultInit()
            }
            for _ in range(self.runs_count):
                alg_builders.extend(partial(b, **deepcopy(common_args)) for b in _builders)
        return alg_builders

    def _get_algorithm_builders_real(self) -> List[Callable[..., Algorithm]]:
        # create builders that create algorithms that work with real number
        # targets; the algorithms are created with the "default" arguments
        return [
            _wire_args_eval(
                partial(
                    BeesAlgorithm,
                    opLocal=RealMutation(delta=0.001),
                    opProbs=UniformPlacesProbs(pscout=0.1),
                ),
                beesNum="popSize",
                opGlobal="opInit or target.defaultInit()",
            ),
            _wire_args_eval(
                partial(
                    BacterialForagingAlgorithm,
                    vel=ExpCool(1.0, 0.35 ** (1 / self.iters_count)),
                    gamma=0.1,
                    probRotate=(0.01, 0.99),
                    mu=0.01,
                    opSelect=TimedOp(
                        ProbOp(ShuffledNeighbors(Tournament(pwin=0.6)), 0.9), 10
                    ),
                    # opDisperse=TimedOp(ProbOp(opInit or target.defaultInit(), 0.01), 20),
                    opSignals=TimedOp(CalcSignals(shape=ShapeClustering(0.00001)), 10),
                ),
                opDisperse="TimedOp(ProbOp(opInit or target.defaultInit(), 0.01), 20)"
            ),
            partial(CompetitiveSwarmOptimizer, socialFactor=0.1, delta=0.01),
            partial(
                DifferentialEvolution,
                opMakeProbe=ProbeClassic(0.8),
                opCrossover=UniformCrossover(0.9),
                opSelect=Tournament(1.0),
            ),
            partial(
                UnivariateMarginalDistributionAlgorithm,
                opSelect=Sorted(SelectLeft(0.7)),
            ),
            partial(
                PopulationBasedIncrementalLearning,
                probMutate=RealMutation(delta=0.01),
                n_best=5,
                n_worst=3,
                p_max=0.95,
                p_min=0.01,
                learning_rate=0.1,
            ),
            partial(
                FireflyAlgorithm,
                alpha=0.01,
                alphabest=0.002,
                betamin=1,
                gamma=0.01,
                theta=1.0,
            ),
            partial(
                GeneticAlgorithm,
                opSelect=ProbOp(ShuffledNeighbors(Tournament(pwin=0.9)), 0.5),
                opCrossover=ShuffledNeighbors(UniformCrossover(pswap=0.3)),
                opMutate=RealMutation(delta=0.01),
            ),
            partial(
                GravitationalSearchAlgorithm,
                g_init=5,
                g_decay_stop_it=self.iters_count,
                alpha=20,
                delta=0.01,
            ),
            partial(
                ParticleSwarmOptimization,
                alphabeta=LinkedAlphaBeta(0.1),
                gamma=0.95,
                delta=0.01,
            ),
            partial(
                SimulatedAnnealing,
                theta=ExpCool(1.0, 0.35 ** (1 / self.iters_count)),
                opMove=RealMutation(delta=0.01),
            ),
        ]

    def _get_algorithm_builders_binary(self) -> List[Callable[..., Algorithm]]:
        raise NotImplementedError

    def _get_algorithm_builders_permutation(self) -> List[Callable[..., Algorithm]]:
        raise NotImplementedError
