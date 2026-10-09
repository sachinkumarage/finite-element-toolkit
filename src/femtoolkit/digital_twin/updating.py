"""Model updating: calibrating parameters against measured data (Version 38).

.. code-block:: text

    Initial parameters
            |
            v
    Run simulation
            |
            v
    Calculate error
            |
            v
    Modify parameters
            |
            v
    Run simulation again
            |
            v
    Calculate new error
            |
            v
    Keep improved parameters
            |
            v
    Repeat

This is a parameter-calibration problem, solved by reusing the *unmodified*
Version 32/33 optimization machinery
(:class:`~femtoolkit.optimization.problems.OptimizationProblem`,
:class:`~femtoolkit.optimization.runner.OptimizationRunner`,
:class:`~femtoolkit.optimization.algorithms.base.StopReason`) -- no second
optimizer is implemented here. The calibration objective is the mean squared
residual

.. math::

    J(\\theta) = \\frac{1}{N} \\sum_{i=1}^{N} \\left(y_{sim,i}(\\theta) - y_{meas,i}\\right)^2

minimized over every *updatable* :class:`~femtoolkit.digital_twin.parameters.ModelParameter`,
each converted directly into a :class:`~femtoolkit.optimization.variables.DesignVariable`
so :math:`\\theta_{min} \\leq \\theta \\leq \\theta_{max}` is enforced by the
same bounded search every other optimization problem in this toolkit uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from femtoolkit.digital_twin.problem import ModelUpdateProblem
from femtoolkit.digital_twin.results import ModelUpdateResult, ModelUpdateStatus
from femtoolkit.optimization.algorithms import OptimizationConfig, StopReason
from femtoolkit.optimization.evaluation import DesignStatus
from femtoolkit.optimization.history import compute_convergence
from femtoolkit.optimization.objectives import Objective, ObjectiveDirection
from femtoolkit.optimization.problems import OptimizationProblem
from femtoolkit.optimization.runner import OptimizationRunner
from femtoolkit.orchestration.simulation import SimulationTask, evaluate_simulation_batch
from femtoolkit.runs.models import RunStatus
from femtoolkit.studies.extractors import get_extractor
from femtoolkit.studies.scenarios import Scenario, apply_scenario

if TYPE_CHECKING:
    from femtoolkit.optimization.context import DesignContext
    from femtoolkit.orchestration.config import OrchestrationConfig

_OBJECTIVE_NAME = "calibration_error"
_PREDICTIONS_METADATA_KEY = "digital_twin_predictions"

_STATUS_FROM_STOP_REASON = {
    StopReason.CONVERGED: ModelUpdateStatus.CONVERGED,
    StopReason.COMPLETED: ModelUpdateStatus.CONVERGED,
    StopReason.TARGET_REACHED: ModelUpdateStatus.CONVERGED,
    StopReason.MAX_EVALUATIONS: ModelUpdateStatus.MAX_EVALUATIONS,
    StopReason.MAX_GENERATIONS: ModelUpdateStatus.MAX_EVALUATIONS,
    StopReason.FAILED: ModelUpdateStatus.FAILED,
    StopReason.CANCELLED: ModelUpdateStatus.FAILED,
}


@dataclass
class ModelUpdateConfig:
    """Configuration for one :func:`run_model_update` call.

    A thin, digital-twin-facing wrapper: internally this is converted
    directly into a Version 32 :class:`~femtoolkit.optimization.algorithms.base.OptimizationConfig`.

    Attributes:
        algorithm: One of :data:`~femtoolkit.optimization.algorithms.SUPPORTED_ALGORITHMS`
            (``"coordinate_search"`` by default -- a transparent, deterministic
            hill-climbing search that matches the modify/re-run/keep-if-improved
            workflow this module's docstring describes).
        max_evaluations: The evaluation budget.
        tolerance: The relative-improvement threshold below which the search
            is considered to have plateaued (see
            :attr:`~femtoolkit.optimization.algorithms.base.StopReason.CONVERGED`).
        patience: How many consecutive evaluations' improvement must stay
            below ``tolerance`` before declaring convergence.
        step_size: For ``"coordinate_search"``, the step taken per move as a
            fraction of a parameter's bound range.
        seed: A random seed; the same seed reproduces the same search
            (subject to the algorithm's own use of randomness --
            ``"coordinate_search"`` is otherwise fully deterministic).
    """

    algorithm: str = "coordinate_search"
    max_evaluations: int = 30
    tolerance: float = 1e-4
    patience: int = 3
    step_size: float = 0.2
    seed: int | None = None

    def to_optimization_config(self) -> OptimizationConfig:
        """Return this configuration as a Version 32 ``OptimizationConfig``."""
        return OptimizationConfig(
            algorithm=self.algorithm,
            max_evaluations=self.max_evaluations,
            tolerance=self.tolerance,
            patience=self.patience,
            step_size=self.step_size,
            seed=self.seed,
        )


@dataclass(frozen=True)
class _CalibrationObjectiveFunction:
    """Picklable objective evaluating :math:`J(\\theta)` at one design point.

    Runs one simulation per measurement point (reusing the unmodified Version
    30 ``Scenario``/``apply_scenario`` and Version 34 ``evaluate_simulation_batch``),
    then records every prediction into ``context.metadata`` so the caller can
    recover the full prediction vector -- not just the aggregated objective --
    without a second round of simulation (spec section 18: avoid unnecessary
    repeated FEA calculations).
    """

    problem: ModelUpdateProblem
    orchestration_config: OrchestrationConfig | None = None

    def __call__(self, context: DesignContext) -> float | None:
        points = self.problem.measurement_data.points
        tasks = [
            SimulationTask(
                task_id=f"calibration-{index}",
                project=apply_scenario(
                    context.project,
                    Scenario(
                        scenario_id=f"calibration-{index}",
                        name=f"Calibration point {index}",
                        parameter_overrides=dict(point.input_conditions),
                    ),
                ),
                scenario_id=f"calibration-{index}",
            )
            for index, point in enumerate(points)
        ]
        runs, _summary = evaluate_simulation_batch(tasks, config=self.orchestration_config)
        extractor = get_extractor(self.problem.measurement_data.quantity_name)

        predictions: list[float] = []
        for run in runs:
            if run.status != RunStatus.COMPLETED:
                return None
            value = extractor(run)
            if value is None:
                return None
            predictions.append(value)

        context.metadata[_PREDICTIONS_METADATA_KEY] = predictions
        measured = [point.measured_value for point in points]
        residuals = np.array(predictions) - np.array(measured)
        return float(np.mean(np.square(residuals)))


def run_model_update(
    problem: ModelUpdateProblem,
    config: ModelUpdateConfig | None = None,
    orchestration_config: OrchestrationConfig | None = None,
) -> ModelUpdateResult:
    """Calibrate every updatable parameter in ``problem`` against its measured data.

    Args:
        problem: The model-updating problem to solve.
        config: The calibration search's configuration. ``None`` uses every
            :class:`ModelUpdateConfig` default.
        orchestration_config: An optional Version 34 orchestration
            configuration for running each evaluation's measurement-point
            batch in parallel. ``None`` runs every point sequentially.

    Returns:
        A :class:`~femtoolkit.digital_twin.results.ModelUpdateResult`.
    """
    config = config or ModelUpdateConfig()
    initial_parameters = problem.current_parameter_values()
    empty = np.array([])

    updatable = problem.updatable_parameters
    if not updatable:
        return ModelUpdateResult(
            initial_parameters=initial_parameters,
            updated_parameters=dict(initial_parameters),
            initial_objective=None,
            final_objective=None,
            initial_predictions=empty,
            updated_predictions=empty,
            measured_values=problem.measurement_data.measured_values(),
            residuals_before=empty,
            residuals_after=empty,
            n_evaluations=0,
            status=ModelUpdateStatus.INVALID,
            stopping_reason="No updatable model parameter was provided.",
        )

    design_variables = [parameter.to_design_variable() for parameter in updatable]
    calibration_function = _CalibrationObjectiveFunction(
        problem=problem, orchestration_config=orchestration_config
    )
    objective = Objective(
        name=_OBJECTIVE_NAME, direction=ObjectiveDirection.MINIMIZE, evaluate=calibration_function
    )
    opt_problem = OptimizationProblem(
        name="Digital twin model update",
        base_project=problem.base_project,
        design_variables=design_variables,
        objectives=[objective],
    )

    opt_result = OptimizationRunner().run(opt_problem, config.to_optimization_config())
    baseline = opt_result.baseline
    measured_values = problem.measurement_data.measured_values()
    convergence_history = [
        step.best_value for step in compute_convergence(opt_result.history, objective)
    ]

    if baseline.status not in (DesignStatus.FEASIBLE, DesignStatus.INFEASIBLE):
        return ModelUpdateResult(
            initial_parameters=initial_parameters,
            updated_parameters=dict(initial_parameters),
            initial_objective=None,
            final_objective=None,
            initial_predictions=empty,
            updated_predictions=empty,
            measured_values=measured_values,
            residuals_before=empty,
            residuals_after=empty,
            n_evaluations=opt_result.history.n_evaluations,
            status=ModelUpdateStatus.FAILED,
            stopping_reason=baseline.error_message or "The baseline simulation did not complete.",
            convergence_history=convergence_history,
        )

    best = opt_result.history.best_feasible(objective) or baseline
    initial_predictions = np.array(baseline.metadata[_PREDICTIONS_METADATA_KEY])
    updated_raw = best.metadata.get(_PREDICTIONS_METADATA_KEY, initial_predictions)
    updated_predictions = np.array(updated_raw)

    updated_parameters = dict(initial_parameters)
    updated_parameters.update(best.design_variables)

    status = _STATUS_FROM_STOP_REASON.get(opt_result.stop_reason, ModelUpdateStatus.MAX_EVALUATIONS)

    return ModelUpdateResult(
        initial_parameters=initial_parameters,
        updated_parameters=updated_parameters,
        initial_objective=baseline.objective_values.get(_OBJECTIVE_NAME),
        final_objective=best.objective_values.get(_OBJECTIVE_NAME),
        initial_predictions=initial_predictions,
        updated_predictions=updated_predictions,
        measured_values=measured_values,
        residuals_before=initial_predictions - measured_values,
        residuals_after=updated_predictions - measured_values,
        n_evaluations=opt_result.history.n_evaluations,
        status=status,
        stopping_reason=opt_result.stop_reason.value,
        convergence_history=convergence_history,
    )


__all__ = ["ModelUpdateConfig", "run_model_update"]
