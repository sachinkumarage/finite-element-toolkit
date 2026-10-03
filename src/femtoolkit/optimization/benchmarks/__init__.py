"""Mathematical optimization benchmarks, independent of FEA (Version 33).

:func:`~femtoolkit.optimization.benchmarks.sphere.sphere`,
:func:`~femtoolkit.optimization.benchmarks.rosenbrock.rosenbrock`, and
:func:`~femtoolkit.optimization.benchmarks.rastrigin.rastrigin` are
plain NumPy functions with a known global optimum, used to verify an
optimization algorithm's mechanics (does it converge toward a known
answer, does it respect bounds, is it reproducible) without the cost or
noise of a real finite element solve standing in for the objective's
mathematical behavior.
:func:`~femtoolkit.optimization.benchmarks.problem_builder.build_benchmark_problem`
is the (separate, clearly-labeled) glue that wraps one of these pure
functions into an
:class:`~femtoolkit.optimization.problems.OptimizationProblem` so it
can be run through the exact same
:class:`~femtoolkit.optimization.runner.OptimizationRunner` and
algorithms as any real engineering problem.
"""

from __future__ import annotations

from femtoolkit.optimization.benchmarks.problem_builder import (
    build_benchmark_problem,
    minimal_benchmark_project,
)
from femtoolkit.optimization.benchmarks.rastrigin import RASTRIGIN_OPTIMUM, rastrigin
from femtoolkit.optimization.benchmarks.rosenbrock import ROSENBROCK_OPTIMUM, rosenbrock
from femtoolkit.optimization.benchmarks.sphere import SPHERE_OPTIMUM, sphere

__all__ = [
    "RASTRIGIN_OPTIMUM",
    "ROSENBROCK_OPTIMUM",
    "SPHERE_OPTIMUM",
    "build_benchmark_problem",
    "minimal_benchmark_project",
    "rastrigin",
    "rosenbrock",
    "sphere",
]
