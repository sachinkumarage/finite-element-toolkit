"""`AdaptiveStudy`: surrogate-assisted adaptive optimization, end to end (Version 36).

.. code-block:: text

    Initial Engineering Samples
            |
            v
    High-Fidelity FEA               (femtoolkit.orchestration, femtoolkit.runs -- V30/34)
            |
            v
    Surrogate Model                 (femtoolkit.surrogate -- V35)
            |
            v
    Candidate Search                (femtoolkit.adaptive.sampling/candidates)
            |
            v
    Candidate Designs
            |
            v
    High-Fidelity Verification      (femtoolkit.surrogate.workflows.verification -- V35)
            |
            v
    Add New Sample / Retrain        (femtoolkit.adaptive.refinement)
            |
            v
    Repeat, until a stopping criterion is met

The surrogate is an acceleration tool, never a replacement for the high-fidelity
model -- see :func:`~femtoolkit.adaptive.study.AdaptiveStudy.run`'s docstring for
exactly how the returned :class:`~femtoolkit.adaptive.results.AdaptiveStudyResult`
keeps the two kinds of result separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from femtoolkit.adaptive.refinement import (
    RefinementConfig,
    SurrogateAcceptanceState,
    run_refinement_step,
)
from femtoolkit.adaptive.results import AdaptiveStudyResult
from femtoolkit.application.project import Project
from femtoolkit.exceptions import InsufficientInitialSamplesError, ValidationError
from femtoolkit.optimization.constraints import Constraint
from femtoolkit.optimization.objectives import Objective, ObjectiveDirection
from femtoolkit.optimization.variables import DesignVariable, validate_unique_variable_names
from femtoolkit.orchestration.config import OrchestrationConfig
from femtoolkit.orchestration.simulation import SimulationTask, evaluate_simulation_batch
from femtoolkit.studies.extractors import Extractor
from femtoolkit.studies.scenarios import Scenario, apply_scenario
from femtoolkit.surrogate.datasets import SnapshotDataset, collect_snapshots_from_runs
from femtoolkit.surrogate.workflows.training import TrainingConfig, train_surrogate

_MINIMUM_INITIAL_SAMPLES = 4
"""The smallest initial sample count a surrogate can realistically be fit on."""


def generate_initial_samples(
    base_project: Project,
    design_variables: list[DesignVariable],
    n_samples: int,
    response_extractors: dict[str, Extractor],
    seed: int | None = None,
    orchestration_config: OrchestrationConfig | None = None,
) -> SnapshotDataset:
    """Build an initial training dataset from ``n_samples`` random design points.

    Unlike :func:`~femtoolkit.surrogate.workflows.training.generate_training_dataset`
    (a Version 30 parameter-sweep Cartesian product), this draws ``n_samples``
    independent points directly from each design variable's domain (see
    :meth:`~femtoolkit.optimization.variables.DesignVariable.sample`) -- appropriate
    for an arbitrary-dimensional design space where a full grid would be wasteful.
    Every point is executed through the same Version 30
    :class:`~femtoolkit.studies.scenarios.Scenario` override mechanism and Version 34
    :func:`~femtoolkit.orchestration.simulation.evaluate_simulation_batch` every other
    high-fidelity batch in this toolkit uses.

    Args:
        base_project: The unmodified base project every sample overrides.
        design_variables: The design variables to sample.
        n_samples: How many initial high-fidelity samples to generate.
        response_extractors: Named result-quantity extractors, keyed by response
            name.
        seed: A random seed; the same seed reproduces the same initial sample points.
        orchestration_config: An optional Version 34 orchestration configuration for
            running the batch in parallel. ``None`` runs every sample sequentially.

    Returns:
        A :class:`~femtoolkit.surrogate.datasets.SnapshotDataset` built from every
        sample that completed successfully.

    Raises:
        ValidationError: If ``design_variables`` is empty or ``n_samples`` is not
            positive.
    """
    if not design_variables:
        raise ValidationError("generate_initial_samples() requires at least one design variable.")
    if n_samples <= 0:
        raise ValidationError(f"n_samples must be positive, got {n_samples}.")
    validate_unique_variable_names(design_variables)

    rng = np.random.default_rng(seed)
    design_points = [
        {variable.path: variable.sample(rng) for variable in design_variables}
        for _ in range(n_samples)
    ]
    tasks = [
        SimulationTask(
            task_id=f"adaptive-initial-{index}",
            project=apply_scenario(
                base_project,
                Scenario(
                    scenario_id=f"adaptive-initial-{index}",
                    name=f"Adaptive initial sample {index}",
                    parameter_overrides=dict(point),
                ),
            ),
            scenario_id=f"adaptive-initial-{index}",
        )
        for index, point in enumerate(design_points)
    ]
    runs, _summary = evaluate_simulation_batch(tasks, config=orchestration_config)
    return collect_snapshots_from_runs(runs, design_points, response_extractors)


@dataclass
class AdaptiveStudy:
    """A surrogate-assisted adaptive engineering study: FEA -> surrogate -> optimize -> verify.

    Attributes:
        base_project: The unmodified base project every design point overrides.
        design_variables: The design variables being searched.
        objective: The objective to minimize/maximize.
        constraints: Constraints every candidate and verification must satisfy.
        response_extractors: Named result-quantity extractors, keyed by response
            name, covering at least ``objective.name`` and every constraint's name.
        model_type: The surrogate model type to train (see
            :data:`~femtoolkit.surrogate.models.SURROGATE_MODEL_TYPES`).
        model_kwargs: Extra keyword arguments forwarded to the surrogate's constructor.
        refinement_config: The adaptive-refinement loop's configuration.
        random_seed: A base random seed for initial sampling and every refinement
            iteration's candidate pool.
    """

    base_project: Project
    design_variables: list[DesignVariable]
    objective: Objective
    constraints: list[Constraint] = field(default_factory=list)
    response_extractors: dict[str, Extractor] = field(default_factory=dict)
    model_type: str = "polynomial"
    model_kwargs: dict[str, object] = field(default_factory=dict)
    refinement_config: RefinementConfig = field(default_factory=RefinementConfig)
    random_seed: int | None = None

    def __post_init__(self) -> None:
        validate_unique_variable_names(self.design_variables)
        if self.objective.name not in self.response_extractors:
            raise ValidationError(
                f"response_extractors must include the objective's response "
                f"{self.objective.name!r}."
            )
        for constraint in self.constraints:
            if constraint.name not in self.response_extractors:
                raise ValidationError(
                    f"response_extractors must include constraint {constraint.name!r}'s response."
                )

    def run(
        self,
        initial_dataset: SnapshotDataset | None = None,
        n_initial_samples: int = 12,
        orchestration_config: OrchestrationConfig | None = None,
    ) -> AdaptiveStudyResult:
        """Run the full adaptive study: build/accept initial samples, then refine.

        Args:
            initial_dataset: An already-collected dataset to start from. ``None``
                generates ``n_initial_samples`` random points with
                :func:`generate_initial_samples`.
            n_initial_samples: How many initial high-fidelity samples to generate
                when ``initial_dataset`` is ``None``.
            orchestration_config: An optional Version 34 orchestration configuration,
                used both for initial sample generation and forwarded to nothing else
                (each refinement iteration verifies exactly one candidate at a time).

        Returns:
            The study's :class:`~femtoolkit.adaptive.results.AdaptiveStudyResult`.

        Raises:
            InsufficientInitialSamplesError: If the initial dataset (generated or
                supplied) has fewer samples than a surrogate can reasonably be fit on.
        """
        if initial_dataset is None:
            dataset = generate_initial_samples(
                self.base_project,
                self.design_variables,
                n_initial_samples,
                self.response_extractors,
                seed=self.random_seed,
                orchestration_config=orchestration_config,
            )
        else:
            dataset = initial_dataset

        if dataset.n_samples < _MINIMUM_INITIAL_SAMPLES:
            raise InsufficientInitialSamplesError(
                f"An adaptive study needs at least {_MINIMUM_INITIAL_SAMPLES} initial "
                f"high-fidelity samples to fit a surrogate; got {dataset.n_samples}."
            )

        initial_sample_count = dataset.n_samples
        training_config = TrainingConfig(
            model_type=self.model_type, model_kwargs=self.model_kwargs, split_seed=self.random_seed
        )
        model, report = train_surrogate(dataset, training_config)

        best_verified_point: dict[str, float] | None = None
        best_verified_objective: float | None = None
        best_predicted_point: dict[str, float] | None = None
        best_predicted_objective: float | None = None
        best_prediction_error: dict[str, float] = {}
        convergence_history: list[float | None] = []
        iteration_history = []
        total_hf_evaluations = initial_sample_count
        surrogate_evaluations = 0
        status = SurrogateAcceptanceState.SURROGATE_ONLY
        stopping_reason = "Completed without any refinement iteration."

        for iteration in range(self.refinement_config.max_iterations):
            hf_budget = self.refinement_config.max_high_fidelity_evaluations
            if total_hf_evaluations - initial_sample_count >= hf_budget:
                stopping_reason = "Reached the maximum number of high-fidelity evaluations."
                break

            dataset, model, step_report, step = run_refinement_step(
                dataset,
                model,
                self.design_variables,
                self.objective,
                self.constraints,
                self.base_project,
                self.response_extractors,
                self.refinement_config,
                iteration,
                training_config=training_config,
            )
            surrogate_evaluations += self.refinement_config.n_candidates
            iteration_history.append(step)
            status = step.status
            if step_report is not None:
                report = step_report

            objective_name = self.objective.name
            direction = self.objective.direction
            predicted_objective = step.candidate.evaluation.objective_values.get(objective_name)
            if predicted_objective is not None and (
                best_predicted_point is None
                or _is_better(predicted_objective, best_predicted_objective, direction)
            ):
                best_predicted_point = dict(step.candidate.point)
                best_predicted_objective = predicted_objective

            if step.verification.actual:
                total_hf_evaluations += 1
                actual_objective = step.verification.actual.get(objective_name)
                if actual_objective is not None and (
                    best_verified_point is None
                    or _is_better(actual_objective, best_verified_objective, direction)
                ):
                    best_verified_point = dict(step.verification.design_point)
                    best_verified_objective = actual_objective
                    best_prediction_error = dict(step.verification.relative_error)

            convergence_history.append(best_verified_objective)

            if _has_converged(convergence_history, self.refinement_config.improvement_tolerance):
                stopping_reason = "Improvement fell below the configured tolerance."
                break
        else:
            stopping_reason = "Reached the maximum number of refinement iterations."

        return AdaptiveStudyResult(
            initial_sample_count=initial_sample_count,
            total_high_fidelity_evaluations=total_hf_evaluations,
            surrogate_evaluations=surrogate_evaluations,
            iteration_count=len(iteration_history),
            best_verified_design=best_verified_point,
            best_verified_objective=best_verified_objective,
            best_surrogate_predicted_design=best_predicted_point,
            best_surrogate_predicted_objective=best_predicted_objective,
            prediction_error=best_prediction_error,
            convergence_history=convergence_history,
            iteration_history=iteration_history,
            final_surrogate_metrics=report,
            status=status,
            stopping_reason=stopping_reason,
        )


def _is_better(candidate: float, incumbent: float | None, direction: ObjectiveDirection) -> bool:
    if incumbent is None:
        return True
    if direction is ObjectiveDirection.MINIMIZE:
        return candidate < incumbent
    return candidate > incumbent


def _has_converged(
    convergence_history: list[float | None], tolerance: float, patience: int = 3
) -> bool:
    recent = [value for value in convergence_history[-patience:] if value is not None]
    if len(recent) < patience:
        return False
    reference = abs(recent[0]) if recent[0] else 1.0
    return all(
        abs(recent[i] - recent[i - 1]) / reference < tolerance for i in range(1, len(recent))
    )


__all__ = ["AdaptiveStudy", "generate_initial_samples"]
