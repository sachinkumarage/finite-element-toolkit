"""`OptimizationRunner`: baseline + algorithm + history -> `OptimizationResult` (Version 32).

.. code-block:: text

    OptimizationProblem + OptimizationConfig
            |
            v
    Evaluate baseline (the problem's default design variable values)
            |
            v
    Run the configured algorithm, recording every evaluation
            |
            v
    OptimizationResult (baseline, history, stop reason)
"""

from __future__ import annotations

from femtoolkit.optimization.algorithms import OptimizationConfig, build_algorithm
from femtoolkit.optimization.evaluation import evaluate_design
from femtoolkit.optimization.history import OptimizationHistory
from femtoolkit.optimization.problems import OptimizationProblem
from femtoolkit.optimization.results import OptimizationResult
from femtoolkit.optimization.robust import RobustDesignConfig, validate_robust_budget
from femtoolkit.runs.manager import SimulationRunManager


class OptimizationRunner:
    """Evaluates a problem's baseline, then runs its configured algorithm."""

    def __init__(self, run_manager: SimulationRunManager | None = None) -> None:
        self._run_manager = run_manager or SimulationRunManager()

    def run(
        self,
        problem: OptimizationProblem,
        config: OptimizationConfig,
        robust_config: RobustDesignConfig | None = None,
    ) -> OptimizationResult:
        """Evaluate the baseline design, then search ``problem``'s design space.

        Args:
            problem: The problem to optimize.
            config: The run's configuration (already validated by its
                own ``__post_init__``).
            robust_config: An optional uncertainty-aware (robust)
                design configuration (Version 33). When
                ``robust_config.uncertainty_enabled`` is ``True``, the
                estimated total FEA count
                (``config.max_evaluations * robust_config.sample_count``)
                is validated against
                ``robust_config.maximum_total_evaluations`` *before*
                the baseline or any other evaluation runs -- see
                :func:`~femtoolkit.optimization.robust.validate_robust_budget`.
                Carried through onto the returned result for
                reproducibility/reporting only; it does not itself
                change how an objective or constraint is evaluated --
                that is determined entirely by whether ``problem``'s
                own objectives/constraints were built with
                :func:`~femtoolkit.optimization.robust.robust_objective_statistic`/
                :func:`~femtoolkit.optimization.robust.robust_constraint_statistic`.

        Returns:
            An :class:`~femtoolkit.optimization.results.OptimizationResult`.

        Raises:
            StudySizeExceededError: If ``robust_config`` is enabled and
                the estimated total FEA count exceeds
                ``robust_config.maximum_total_evaluations``.
        """
        validate_robust_budget(config.max_evaluations, robust_config)

        baseline = evaluate_design(
            design_id=f"{problem.name}-baseline",
            values=problem.default_values(),
            design_variables=problem.design_variables,
            base_project=problem.base_project,
            objectives=problem.objectives,
            constraints=problem.constraints,
            run_manager=self._run_manager,
        )

        history = OptimizationHistory()
        algorithm = build_algorithm(config.algorithm)
        stop_reason = algorithm.optimize(
            problem, config, self._run_manager, history, starting_evaluation=baseline
        )

        return OptimizationResult(
            problem_name=problem.name,
            config=config,
            design_variables=list(problem.design_variables),
            objectives=list(problem.objectives),
            baseline=baseline,
            history=history,
            stop_reason=stop_reason,
            robust_config=robust_config,
        )


__all__ = ["OptimizationRunner"]
