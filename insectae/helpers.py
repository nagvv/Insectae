from types import MethodType
from typing import Tuple, Union, Callable

from .executor import BaseExecutor
from .goals import Goal, ToMax, ToMin
from .metrics import Metrics
from .targets import BinaryTarget, RealTarget


def from_ioh_problem(
    problem, verbose: int = 0
) -> Tuple[Union[RealTarget, BinaryTarget], Goal, Metrics]:
    """Create insectae classes from a IOHexperimenter problem instance.

    Creates `Target`, `Goal` and `Metrics` instances from a provided
    IOHexperimenter problem instance. The `verbose` variable is forwarded to
    `Metrics` class constructor.

    Parameters
    ----------
    `problem` : RealSingleObjective or IntegerSingleObjective
    `verbose` : int
        verbose value to forward to a metrics instance

    Returns
    -------
    tuple(`RealTarget` or `BinaryTarget`, `Goal`, `Metrics`)

    Raises
    ------
    `ImportError`
        If ioh can not be imported.
    `TypeError`
        If unsupported type of a problem is provided.
    `ValueError`
        If unable to parse a problem
    """
    from ioh import problem as ioh_problem
    from ioh.iohcpp import OptimizationType

    def get_goal(opt_type: OptimizationType) -> Goal:
        if opt_type == OptimizationType.MIN:
            return ToMin()
        elif opt_type == OptimizationType.MAX:
            return ToMax()
        else:
            raise ValueError("unknown optimization type")

    if isinstance(problem, ioh_problem.RealSingleObjective):
        goal = get_goal(problem.meta_data.optimization_type)
        metrics = Metrics(goal=goal, verbose=verbose)
        target = RealTarget(
            target=problem,
            dimension=problem.meta_data.n_variables,
            metrics=metrics,
            bounds=(problem.bounds.lb, problem.bounds.ub),
        )
    elif isinstance(problem, ioh_problem.IntegerSingleObjective):
        goal = get_goal(problem.meta_data.optimization_type)
        metrics = Metrics(goal=goal, verbose=verbose)
        target = BinaryTarget(
            target=problem, dimension=problem.meta_data.n_variables, metrics=metrics
        )
    else:
        raise TypeError("unsupported type of a problem")

    return target, goal, metrics


def wrap_executor_evaluate(
    executor: BaseExecutor,
    new_evaluate: Callable,
    orig_evaluate_field: str = "_orig_evaluate"
):
    setattr(executor, orig_evaluate_field, executor.evaluate)
    executor.evaluate = MethodType(new_evaluate, executor)

    def restore():
        executor.evaluate = getattr(executor, orig_evaluate_field)
        delattr(executor, orig_evaluate_field)

    return restore
