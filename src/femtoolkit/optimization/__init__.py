"""Engineering optimization and design-space exploration (Version 32).

Built entirely on top of the existing simulation-study infrastructure:
a design's variable values become a
:class:`~femtoolkit.studies.scenarios.Scenario` the same way a Version
30 parameter-sweep value or a Version 31 Monte Carlo sample does, and
every evaluation is executed by the unmodified
:class:`~femtoolkit.runs.manager.SimulationRunManager` -- there is
exactly one simulation-execution system in this toolkit, not two.

.. code-block:: text

    Design Variables (femtoolkit.optimization.variables)
            |
            v
    Objectives + Constraints (femtoolkit.optimization.objectives / .constraints)
            |
            v
    OptimizationProblem (femtoolkit.optimization.problems)
            |
            v
    OptimizationAlgorithm: random search / coordinate search
    (femtoolkit.optimization.algorithms)
            |
            v
    OptimizationHistory -> OptimizationResult
    (femtoolkit.optimization.history / .results)
            |
            v
    Pareto front (multi-objective) / report / plots
    (femtoolkit.optimization.pareto / .report / .plots)

**Transparency over sophistication.** Both implemented algorithms
(bounded random search, coordinate search) are simple enough that a
reader can trace exactly why the search moved from one design to the
next. Neither claims to find a global optimum, and this package never
automatically selects a single "best" solution from a multi-objective
Pareto front -- see ``docs/optimization.md`` for the full guide and
explicit scope boundaries.

**Explicit scope exclusions (this version).** No machine learning, no
surrogate/Gaussian-process models, no Bayesian optimization, no
topology or gradient-based/adjoint optimization, no genetic algorithms,
no particle swarm optimization, no distributed/GPU optimization, and no
full robust-optimization algorithm (a single lightweight, optional
building block toward one is provided --
:func:`~femtoolkit.optimization.objectives.robust_objective_mean`).
See the Version 33 preview in the main README for the planned next
direction.
"""

from __future__ import annotations

from femtoolkit.optimization.algorithms import (
    DEFAULT_EVALUATION_LIMIT,
    DEFAULT_MAX_EVALUATIONS,
    SUPPORTED_ALGORITHMS,
    BoundedRandomSearch,
    CoordinateSearch,
    OptimizationAlgorithm,
    OptimizationConfig,
    StopReason,
)
from femtoolkit.optimization.constraints import (
    DEFAULT_CONSTRAINT_TOLERANCE,
    Constraint,
    ConstraintRelation,
)
from femtoolkit.optimization.context import DesignContext
from femtoolkit.optimization.evaluation import (
    ConstraintEvaluation,
    DesignEvaluation,
    DesignStatus,
    evaluate_design,
    feasibility_rank,
    is_better_evaluation,
)
from femtoolkit.optimization.history import (
    ConvergenceStep,
    OptimizationHistory,
    compute_convergence,
)
from femtoolkit.optimization.objectives import (
    Objective,
    ObjectiveDirection,
    from_result_extractor,
    rectangular_mass,
    robust_objective_mean,
)
from femtoolkit.optimization.pareto import dominates, pareto_front
from femtoolkit.optimization.plots import (
    plot_constraint_violation_history,
    plot_design_variable_history,
    plot_objective_history,
    plot_pareto_front,
)
from femtoolkit.optimization.problems import OptimizationMode, OptimizationProblem
from femtoolkit.optimization.report import (
    OptimizationReport,
    build_optimization_report,
    render_optimization_report_html,
    render_optimization_report_markdown,
    save_optimization_report,
)
from femtoolkit.optimization.results import OptimizationResult
from femtoolkit.optimization.runner import OptimizationRunner
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType

__all__ = [
    "DEFAULT_CONSTRAINT_TOLERANCE",
    "DEFAULT_EVALUATION_LIMIT",
    "DEFAULT_MAX_EVALUATIONS",
    "SUPPORTED_ALGORITHMS",
    "BoundedRandomSearch",
    "Constraint",
    "ConstraintEvaluation",
    "ConstraintRelation",
    "ConvergenceStep",
    "CoordinateSearch",
    "DesignContext",
    "DesignEvaluation",
    "DesignStatus",
    "DesignVariable",
    "DesignVariableType",
    "Objective",
    "ObjectiveDirection",
    "OptimizationAlgorithm",
    "OptimizationConfig",
    "OptimizationHistory",
    "OptimizationMode",
    "OptimizationProblem",
    "OptimizationReport",
    "OptimizationResult",
    "OptimizationRunner",
    "StopReason",
    "build_optimization_report",
    "compute_convergence",
    "dominates",
    "evaluate_design",
    "feasibility_rank",
    "from_result_extractor",
    "is_better_evaluation",
    "pareto_front",
    "plot_constraint_violation_history",
    "plot_design_variable_history",
    "plot_objective_history",
    "plot_pareto_front",
    "rectangular_mass",
    "render_optimization_report_html",
    "render_optimization_report_markdown",
    "robust_objective_mean",
    "save_optimization_report",
]
