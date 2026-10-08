"""Adaptive refinement: add one high-fidelity sample, retrain, repeat (Version 36).

.. code-block:: text

    Surrogate
        |
        v
    Candidate search (femtoolkit.adaptive.sampling)
        |
        v
    Best candidate
        |
        v
    High-fidelity FEA verification (femtoolkit.surrogate.workflows.verification)
        |
        v
    Add sample to dataset (femtoolkit.surrogate.datasets.SnapshotDataset.with_additional_snapshots)
        |
        v
    Retrain surrogate (femtoolkit.surrogate.workflows.training.train_surrogate)
        |
        v
    Repeat, until a stopping criterion is met

Every high-fidelity evaluation here goes through
:func:`~femtoolkit.surrogate.workflows.verification.verify_against_high_fidelity` --
the surrogate's own candidate ranking never substitutes for an actual FEA solve; it
only decides *where* the next solve should be spent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from femtoolkit.adaptive.candidates import generate_candidate_pool
from femtoolkit.adaptive.sampling import CandidateScore, SamplingStrategy, rank_candidates
from femtoolkit.application.project import Project
from femtoolkit.exceptions import ValidationError
from femtoolkit.optimization.constraints import Constraint
from femtoolkit.optimization.objectives import Objective, ObjectiveDirection
from femtoolkit.optimization.variables import DesignVariable
from femtoolkit.orchestration.random_state import derive_task_seed
from femtoolkit.studies.extractors import Extractor
from femtoolkit.surrogate.datasets import Snapshot, SnapshotDataset
from femtoolkit.surrogate.models.base import SurrogateModel
from femtoolkit.surrogate.validation import EngineeringTolerance, SurrogateValidationReport
from femtoolkit.surrogate.workflows.evaluator import SurrogateEvaluator
from femtoolkit.surrogate.workflows.training import TrainingConfig, train_surrogate
from femtoolkit.surrogate.workflows.verification import (
    AcceptanceStatus,
    SurrogateVerificationRecord,
    verify_against_high_fidelity,
)


class SurrogateAcceptanceState(Enum):
    """Engineering status of a surrogate-derived result (spec section 17).

    Attributes:
        SURROGATE_ONLY: Only a surrogate prediction exists; no high-fidelity
            verification has been performed for this design.
        PENDING_VERIFICATION: A high-fidelity verification has been requested but not
            yet completed.
        VERIFIED: A high-fidelity result confirms the surrogate's prediction within
            the configured error tolerance.
        VERIFICATION_FAILED: The high-fidelity verification run itself did not
            complete; no comparison could be made.
        REQUIRES_REFINEMENT: The high-fidelity result is available but disagrees with
            the surrogate's prediction beyond the configured error tolerance.
    """

    SURROGATE_ONLY = "surrogate_only"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    VERIFICATION_FAILED = "verification_failed"
    REQUIRES_REFINEMENT = "requires_refinement"


@dataclass
class RefinementConfig:
    """Configuration for one or more :func:`run_refinement_step` calls.

    Attributes:
        max_iterations: The maximum number of refinement iterations an
            :class:`~femtoolkit.adaptive.study.AdaptiveStudy` may run.
        max_high_fidelity_evaluations: The maximum number of *additional*
            high-fidelity evaluations an adaptive study may perform beyond its
            initial samples.
        error_tolerance: The maximum acceptable relative error (spec section 11)
            before a verified result is flagged
            :attr:`SurrogateAcceptanceState.REQUIRES_REFINEMENT`.
        improvement_tolerance: The minimum relative improvement in the best verified
            objective value an iteration must achieve to avoid triggering an
            early-stop on stalled progress.
        n_candidates: How many candidate points :func:`run_refinement_step` scores
            per iteration (see :func:`~femtoolkit.adaptive.candidates.generate_candidate_pool`).
        sampling_strategy: Which :class:`~femtoolkit.adaptive.sampling.SamplingStrategy`
            ranks the candidate pool.
        exploration_weight: The hybrid strategy's exploration weight.
        exploitation_weight: The hybrid strategy's exploitation weight.
        tolerances: Engineering tolerances checked against every high-fidelity
            verification (forwarded to
            :func:`~femtoolkit.surrogate.workflows.verification.verify_against_high_fidelity`).
        seed: A base random seed; each iteration derives its own candidate-pool seed
            from this deterministically (see
            :func:`~femtoolkit.orchestration.random_state.derive_task_seed`), so the
            same seed reproduces the same sequence of refinement steps.
    """

    max_iterations: int = 10
    max_high_fidelity_evaluations: int = 20
    error_tolerance: float = 0.1
    improvement_tolerance: float = 1e-3
    n_candidates: int = 50
    sampling_strategy: SamplingStrategy = SamplingStrategy.HYBRID
    exploration_weight: float = 0.5
    exploitation_weight: float = 0.5
    tolerances: list[EngineeringTolerance] = field(default_factory=list)
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValidationError(f"max_iterations must be positive, got {self.max_iterations}.")
        if self.max_high_fidelity_evaluations <= 0:
            raise ValidationError(
                f"max_high_fidelity_evaluations must be positive, got "
                f"{self.max_high_fidelity_evaluations}."
            )
        if self.error_tolerance <= 0.0:
            raise ValidationError(f"error_tolerance must be positive, got {self.error_tolerance}.")
        if self.n_candidates <= 0:
            raise ValidationError(f"n_candidates must be positive, got {self.n_candidates}.")


@dataclass(frozen=True)
class RefinementStepResult:
    """The outcome of one :func:`run_refinement_step` call.

    Attributes:
        iteration: This step's iteration index (0-based).
        candidate: The scored candidate that was selected for high-fidelity
            evaluation.
        verification: The high-fidelity verification record for that candidate.
        status: This step's :class:`SurrogateAcceptanceState`.
        dataset_version: The training dataset's version after this step (unchanged
            from before the step if the high-fidelity run failed and no sample could
            be added).
    """

    iteration: int
    candidate: CandidateScore
    verification: SurrogateVerificationRecord
    status: SurrogateAcceptanceState
    dataset_version: int


def prediction_agreement(
    actual_center: float,
    actual_new: float,
    predicted_center: float,
    predicted_new: float,
    direction: ObjectiveDirection = ObjectiveDirection.MINIMIZE,
) -> float | None:
    """The agreement ratio between predicted and actual improvement (spec section 14).

    .. math::

        \\rho = \\frac{f(x_c) - f(x_{new})}{\\hat f(x_c) - \\hat f(x_{new})}

    (sign-flipped for a maximization objective). Used only as an engineering
    diagnostic for whether the surrogate is behaving reliably near the current design
    center -- not as a formal trust-region optimality criterion.

    Args:
        actual_center: The high-fidelity objective value at the trust region's center.
        actual_new: The high-fidelity objective value at the new candidate.
        predicted_center: The surrogate's predicted objective value at the center.
        predicted_new: The surrogate's predicted objective value at the new candidate.
        direction: Whether the objective is minimized or maximized.

    Returns:
        The agreement ratio ``rho``, or ``None`` if the predicted improvement is
        too close to zero for the ratio to be meaningful (within ``1e-12``).
    """
    sign = -1.0 if direction is ObjectiveDirection.MINIMIZE else 1.0
    actual_improvement = sign * (actual_center - actual_new)
    predicted_improvement = sign * (predicted_center - predicted_new)
    if abs(predicted_improvement) < 1e-12:
        return None
    return actual_improvement / predicted_improvement


def _acceptance_status(
    record: SurrogateVerificationRecord, error_tolerance: float
) -> SurrogateAcceptanceState:
    if record.acceptance is AcceptanceStatus.FAILED:
        return SurrogateAcceptanceState.VERIFICATION_FAILED
    if record.acceptance is AcceptanceStatus.REJECT:
        return SurrogateAcceptanceState.REQUIRES_REFINEMENT
    if record.relative_error:
        worst_relative_error = max(record.relative_error.values())
        if worst_relative_error > error_tolerance:
            return SurrogateAcceptanceState.REQUIRES_REFINEMENT
    return SurrogateAcceptanceState.VERIFIED


def run_refinement_step(
    dataset: SnapshotDataset,
    model: SurrogateModel,
    design_variables: list[DesignVariable],
    objective: Objective,
    constraints: list[Constraint],
    base_project: Project,
    response_extractors: dict[str, Extractor],
    config: RefinementConfig,
    iteration: int,
    training_config: TrainingConfig | None = None,
) -> tuple[SnapshotDataset, SurrogateModel, SurrogateValidationReport | None, RefinementStepResult]:
    """Run one adaptive-refinement iteration.

    Generates a candidate pool, ranks it with the current surrogate, verifies the
    top-ranked candidate against real high-fidelity FEA, and -- if that run
    completed -- adds the new sample to the dataset and retrains the surrogate.

    Args:
        dataset: The current training dataset.
        model: The current, already-fitted surrogate model.
        design_variables: The design variables being searched.
        objective: The objective driving candidate ranking.
        constraints: Constraints evaluated alongside the objective (predicted by the
            same ``model`` as the objective -- see
            :class:`~femtoolkit.surrogate.workflows.evaluator.SurrogateEvaluator`).
        base_project: The unmodified base project the selected candidate overrides
            for high-fidelity verification.
        response_extractors: Named result-quantity extractors, keyed by response
            name, covering at least ``objective.name`` and every constraint's name.
        config: This refinement run's configuration.
        iteration: This step's iteration index (0-based; used to derive a
            reproducible per-step candidate seed from ``config.seed``).
        training_config: The configuration to retrain the surrogate with. ``None``
            reuses ``model``'s own type and scaling configuration with
            ``config.seed``.

    Returns:
        A ``(dataset, model, report, step)`` tuple: the dataset and model to use for
        the next iteration (unchanged from the inputs if the high-fidelity run
        failed), the new validation report (``None`` if no retraining occurred), and
        this step's :class:`RefinementStepResult`.

    Raises:
        ValidationError: If no candidate in the generated pool could be scored.
    """
    surrogates: dict[str, SurrogateModel] = {objective.name: model}
    for constraint in constraints:
        surrogates.setdefault(constraint.name, model)
    evaluator = SurrogateEvaluator(surrogates)

    pool_seed = derive_task_seed(config.seed, f"adaptive-refinement-{iteration}")
    candidates = generate_candidate_pool(design_variables, config.n_candidates, seed=pool_seed)
    ranked = rank_candidates(
        candidates,
        dataset,
        design_variables,
        evaluator,
        objective,
        strategy=config.sampling_strategy,
        exploration_weight=config.exploration_weight,
        exploitation_weight=config.exploitation_weight,
    )
    if not ranked:
        raise ValidationError(
            "run_refinement_step() could not score any candidate -- every generated "
            "candidate was either invalid or could not be predicted by the surrogate."
        )
    best = ranked[0]

    path_by_name = {variable.name: variable.path for variable in design_variables}
    design_point = {path_by_name[name]: value for name, value in best.point.items()}

    records = verify_against_high_fidelity(
        model, base_project, [design_point], response_extractors, tolerances=config.tolerances
    )
    record = records[0]
    status = _acceptance_status(record, config.error_tolerance)

    if not record.actual:
        step = RefinementStepResult(
            iteration=iteration,
            candidate=best,
            verification=record,
            status=status,
            dataset_version=dataset.dataset_version,
        )
        return dataset, model, None, step

    new_snapshot = Snapshot(
        snapshot_id=f"adaptive-{iteration}", inputs=dict(design_point), outputs=dict(record.actual)
    )
    updated_dataset = dataset.with_additional_snapshots([new_snapshot])
    retrain_config = training_config or TrainingConfig(
        model_type=model.name, split_seed=config.seed
    )
    new_model, report = train_surrogate(updated_dataset, retrain_config)

    step = RefinementStepResult(
        iteration=iteration,
        candidate=best,
        verification=record,
        status=status,
        dataset_version=updated_dataset.dataset_version,
    )
    return updated_dataset, new_model, report, step


__all__ = [
    "RefinementConfig",
    "RefinementStepResult",
    "SurrogateAcceptanceState",
    "prediction_agreement",
    "run_refinement_step",
]
