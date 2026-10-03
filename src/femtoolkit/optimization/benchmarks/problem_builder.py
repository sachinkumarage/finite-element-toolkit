"""Glue between a pure-math benchmark function and the optimization framework (Version 33).

:mod:`.sphere`, :mod:`.rosenbrock`, and :mod:`.rastrigin` are plain
NumPy functions with zero finite element dependency. This module is
the one place that wraps one of them into an
:class:`~femtoolkit.optimization.problems.OptimizationProblem` so
:class:`~femtoolkit.optimization.runner.OptimizationRunner` -- and
therefore every algorithm in :mod:`femtoolkit.optimization.algorithms`
-- can be exercised end to end against a benchmark with a known
optimum, exactly like :mod:`femtoolkit.optimization` does for a real
engineering model. This toolkit has exactly one evaluation pipeline
(design variables -> scenario -> FEA run -> objective), so even a
"pure math" benchmark still triggers one minimal, inexpensive, always-
solvable FEA run per evaluation; the benchmark function itself never
touches the FEA result.

Each design variable is mapped to one entry of the base project's
``loads`` list (``loads.<i>.magnitude``) rather than a geometric or
material field, since a nodal load magnitude accepts any finite real
number -- unlike thickness or Young's modulus, it needs no positivity
constraint that would otherwise clip a benchmark's natural domain
(e.g. Rastrigin's standard ``[-5.12, 5.12]``).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.exceptions import ValidationError
from femtoolkit.optimization.objectives import Objective, ObjectiveDirection
from femtoolkit.optimization.problems import OptimizationProblem
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType


def minimal_benchmark_project(n_dimensions: int) -> Project:
    """A tiny, always-solvable project with ``n_dimensions`` independent load entries.

    Args:
        n_dimensions: How many design variables the benchmark needs;
            one load entry is created per dimension so each can be
            overridden independently via ``loads.<i>.magnitude``.

    Returns:
        A minimal, valid cantilever-style :class:`~femtoolkit.application.project.Project`.
    """
    project = Project(name="benchmark", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.material.density = 7850.0
    project.mesh.width, project.mesh.height = 1.0, 0.2
    project.mesh.nx, project.mesh.ny = 2, 1
    project.mesh.thickness = 0.01
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [
        LoadConfig(region="right", dof="Y", magnitude=-100.0) for _ in range(n_dimensions)
    ]
    return project


def build_benchmark_problem(
    name: str,
    function: Callable[[np.ndarray], float],
    n_dimensions: int,
    lower_bound: float,
    upper_bound: float,
    objective_name: str = "f",
) -> OptimizationProblem:
    """Wrap a pure-math benchmark function as a single-objective, unconstrained problem.

    Args:
        name: The resulting problem's name.
        function: A benchmark function, e.g.
            :func:`~femtoolkit.optimization.benchmarks.sphere.sphere`.
        n_dimensions: How many design variables to create (must be at
            least 1, and at least 2 for
            :func:`~femtoolkit.optimization.benchmarks.rosenbrock.rosenbrock`).
        lower_bound: The shared lower bound for every design variable.
        upper_bound: The shared upper bound for every design variable.
        objective_name: The resulting objective's name.

    Returns:
        An :class:`~femtoolkit.optimization.problems.OptimizationProblem`
        ready to hand to :class:`~femtoolkit.optimization.runner.OptimizationRunner`.

    Raises:
        ValidationError: If ``n_dimensions`` is less than 1.
    """
    if n_dimensions < 1:
        raise ValidationError(f"n_dimensions must be at least 1, got {n_dimensions}.")

    project = minimal_benchmark_project(n_dimensions)
    variables = [
        DesignVariable(
            name=f"x{index}",
            path=f"loads.{index}.magnitude",
            variable_type=DesignVariableType.CONTINUOUS,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            default_value=(lower_bound + upper_bound) / 2.0,
        )
        for index in range(n_dimensions)
    ]
    variable_names = [variable.name for variable in variables]

    def _evaluate(context) -> float:  # noqa: ANN001
        vector = np.array([context.design_variables[name] for name in variable_names])
        return function(vector)

    objective = Objective(
        name=objective_name, direction=ObjectiveDirection.MINIMIZE, evaluate=_evaluate
    )
    return OptimizationProblem(
        name=name,
        base_project=project,
        design_variables=variables,
        objectives=[objective],
        description=f"Mathematical benchmark problem wrapping {function.__name__!r}.",
    )


__all__ = ["build_benchmark_problem", "minimal_benchmark_project"]
