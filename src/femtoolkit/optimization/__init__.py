"""Engineering optimization and design-space exploration (Version 32/33).

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
    (optionally uncertainty-aware -- femtoolkit.optimization.robust)
            |
            v
    OptimizationProblem (femtoolkit.optimization.problems)
            |
            v
    OptimizationAlgorithm: random search / coordinate search /
    differential evolution / genetic algorithm / particle swarm / NSGA-II
    (femtoolkit.optimization.algorithms)
            |
            v
    OptimizationHistory -> OptimizationResult
    (femtoolkit.optimization.history / .results)
            |
            v
    Pareto front (multi-objective) / report / plots
    (femtoolkit.optimization.pareto / .report / .plots)

**Transparency over sophistication.** The two Version 32 single-point
algorithms (bounded random search, coordinate search) are simple enough
that a reader can trace exactly why the search moved from one design to
the next. The four Version 33 population-based algorithms (differential
evolution, genetic algorithm, particle swarm optimization, NSGA-II) add
exploration/exploitation dynamics a single point cannot express, while
reusing the exact same evaluation pipeline, feasibility-first
comparison, and :class:`~femtoolkit.optimization.algorithms.base.OptimizationAlgorithm`
interface. None of the six claims to find a global optimum, and this
package never automatically selects a single "best" solution from a
multi-objective Pareto front -- see ``docs/optimization.md`` for the
full guide and explicit scope boundaries.

**Robust (uncertainty-aware) design (Version 33).** Any objective or
constraint can optionally be built from
:func:`~femtoolkit.optimization.robust.robust_objective_statistic`/
:func:`~femtoolkit.optimization.robust.robust_constraint_statistic`,
which evaluate a statistic (mean, percentile, exceedance probability, ...)
of a quantity over a small Version 31 Monte Carlo study at each design
point instead of one deterministic value. This is strictly opt-in
(:class:`~femtoolkit.optimization.robust.RobustDesignConfig.uncertainty_enabled`)
and guarded by an explicit, validated evaluation-budget ceiling before
any simulation runs.

**Explicit scope exclusions (this version).** No machine learning, no
surrogate/Gaussian-process models, no Bayesian optimization, no
topology or gradient-based/adjoint optimization, no distributed/GPU
optimization, no FORM/SORM or other rigorous reliability method, no
stochastic finite elements, and no commercial-solver integration. See
the Version 34 preview in the main README for the planned next
direction.
"""

from __future__ import annotations

from femtoolkit.optimization.algorithms import (
    DEFAULT_EVALUATION_LIMIT,
    DEFAULT_MAX_EVALUATIONS,
    DEFAULT_MAX_GENERATIONS,
    NSGA2,
    SUPPORTED_ALGORITHMS,
    BoundedRandomSearch,
    CoordinateSearch,
    DifferentialEvolution,
    GeneticAlgorithm,
    OptimizationAlgorithm,
    OptimizationConfig,
    ParticleSwarmOptimization,
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
    GenerationSummary,
    OptimizationHistory,
    compute_convergence,
    generation_summaries,
)
from femtoolkit.optimization.objectives import (
    Objective,
    ObjectiveDirection,
    from_result_extractor,
    rectangular_mass,
    robust_objective_mean,
)
from femtoolkit.optimization.pareto import (
    constrained_dominates,
    crowding_distance,
    dominates,
    fast_non_dominated_sort,
    pareto_front,
)
from femtoolkit.optimization.plots import (
    plot_constraint_violation_history,
    plot_design_variable_history,
    plot_generation_objective_history,
    plot_objective_history,
    plot_pareto_front,
    plot_pareto_front_3d,
    plot_pareto_front_size_history,
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
from femtoolkit.optimization.robust import (
    DEFAULT_MAXIMUM_TOTAL_EVALUATIONS,
    DEFAULT_SAMPLE_COUNT,
    SUPPORTED_STATISTICS,
    RobustDesignConfig,
    estimate_total_fea_count,
    robust_constraint_statistic,
    robust_objective_statistic,
    validate_robust_budget,
)
from femtoolkit.optimization.runner import OptimizationRunner
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType

__all__ = [
    "DEFAULT_CONSTRAINT_TOLERANCE",
    "DEFAULT_EVALUATION_LIMIT",
    "DEFAULT_MAXIMUM_TOTAL_EVALUATIONS",
    "DEFAULT_MAX_EVALUATIONS",
    "DEFAULT_MAX_GENERATIONS",
    "DEFAULT_SAMPLE_COUNT",
    "SUPPORTED_ALGORITHMS",
    "SUPPORTED_STATISTICS",
    "NSGA2",
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
    "DifferentialEvolution",
    "GenerationSummary",
    "GeneticAlgorithm",
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
    "ParticleSwarmOptimization",
    "RobustDesignConfig",
    "StopReason",
    "build_optimization_report",
    "compute_convergence",
    "constrained_dominates",
    "crowding_distance",
    "dominates",
    "estimate_total_fea_count",
    "evaluate_design",
    "fast_non_dominated_sort",
    "feasibility_rank",
    "from_result_extractor",
    "generation_summaries",
    "is_better_evaluation",
    "pareto_front",
    "plot_constraint_violation_history",
    "plot_design_variable_history",
    "plot_generation_objective_history",
    "plot_objective_history",
    "plot_pareto_front",
    "plot_pareto_front_3d",
    "plot_pareto_front_size_history",
    "rectangular_mass",
    "render_optimization_report_html",
    "render_optimization_report_markdown",
    "robust_constraint_statistic",
    "robust_objective_mean",
    "robust_objective_statistic",
    "save_optimization_report",
    "validate_robust_budget",
]
