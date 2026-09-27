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
from femtoolkit.runs.manager import SimulationRunManager


class OptimizationRunner:
    """Evaluates a problem's baseline, then runs its configured algorithm."""

    def __init__(self, run_manager: SimulationRunManager | None = None) -> None:
        self._run_manager = run_manager or SimulationRunManager()

    def run(self, problem: OptimizationProblem, config: OptimizationConfig) -> OptimizationResult:
        """Evaluate the baseline design, then search ``problem``'s design space.

        Args:
            problem: The problem to optimize.
            config: The run's configuration (already validated by its
                own ``__post_init__``).

        Returns:
            An :class:`~femtoolkit.optimization.results.OptimizationResult`.
        """
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
        )


__all__ = ["OptimizationRunner"]
