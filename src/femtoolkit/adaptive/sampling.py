"""Adaptive sampling: deciding where the next expensive FEA evaluation should go (Version 36).

**Engineering concept.** Once a surrogate exists (:mod:`femtoolkit.surrogate`), the
question is no longer "approximate or exact?" but "where is the next high-fidelity
evaluation worth its cost?" This module scores a pool of candidate design points with
one of four deliberately transparent, heuristic strategies -- never a black-box
acquisition function:

- **Distance-based exploration** -- prefer points far (in normalized design space)
  from every existing training sample, so the surrogate learns about poorly covered
  regions.
- **Error-based refinement** -- prefer points the surrogate itself is least confident
  about, using the Version 35 applicability-domain status
  (:class:`~femtoolkit.surrogate.domain.DomainStatus`) as a transparent proxy for
  "surrogate error is likely high here" (an
  :attr:`~femtoolkit.surrogate.domain.DomainStatus.OUTSIDE_TRAINING_DOMAIN` point
  scores higher than a :attr:`~femtoolkit.surrogate.domain.DomainStatus.BOUNDARY` one,
  which scores higher than one safely
  :attr:`~femtoolkit.surrogate.domain.DomainStatus.WITHIN_TRAINING_DOMAIN`).
- **Objective-based refinement** -- prefer points the surrogate predicts are good
  designs, ranked with the same feasibility-first comparison
  (:func:`~femtoolkit.optimization.evaluation.is_better_evaluation`) every Version 32/33
  optimization algorithm already uses.
- **Hybrid** -- a weighted combination,

  .. math::

      Score(x) = w_e E(x) + w_o O(x)

  where :math:`E(x)` is the exploration score, :math:`O(x)` is the objective
  (exploitation) score, and :math:`w_e`/:math:`w_o` are caller-configured weights. This
  is an engineering heuristic, not a mathematically universal acquisition function.

**Exploration vs. exploitation.** Exploration searches poorly sampled regions of the
design space (reduces the surrogate's own uncertainty); exploitation searches regions
the current surrogate predicts are promising (pursues the optimization objective). A
study that only exploits can get stuck refining an already-well-sampled local optimum;
a study that only explores never converges. The hybrid weights let a caller trade one
off against the other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from femtoolkit.optimization.evaluation import DesignEvaluation, DesignStatus, is_better_evaluation
from femtoolkit.optimization.objectives import Objective
from femtoolkit.optimization.variables import DesignVariable
from femtoolkit.surrogate.datasets import SnapshotDataset
from femtoolkit.surrogate.domain import DomainStatus
from femtoolkit.surrogate.workflows.evaluator import SurrogateEvaluator

_DOMAIN_ERROR_PROXY = {
    DomainStatus.WITHIN_TRAINING_DOMAIN: 0.0,
    DomainStatus.BOUNDARY: 0.5,
    DomainStatus.OUTSIDE_TRAINING_DOMAIN: 1.0,
    DomainStatus.INVALID: 1.0,
}
"""How much Version 35 applicability-domain status counts as an "error-based" priority."""


class SamplingStrategy(Enum):
    """Which heuristic :func:`rank_candidates` uses to score a candidate pool.

    Attributes:
        DISTANCE: Strategy A -- pure exploration, farthest-from-training-data first.
        ERROR: Strategy B -- pure error-based refinement, using applicability-domain
            status as a transparent proxy for expected surrogate error.
        OBJECTIVE: Strategy C -- pure exploitation, best-predicted-objective first.
        HYBRID: Strategy D -- a weighted combination of exploration and objective
            scores (spec section 8).
    """

    DISTANCE = "distance"
    ERROR = "error"
    OBJECTIVE = "objective"
    HYBRID = "hybrid"


def design_variable_bounds(
    design_variables: list[DesignVariable],
) -> dict[str, tuple[float, float]]:
    """Return each continuous/integer design variable's ``(lower, upper)`` bound, by path.

    Categorical variables have no numeric bound and are omitted -- the normalized
    distance/exploration metrics in this module are defined only over the continuous
    design space (see :func:`normalized_distance`'s docstring).

    Args:
        design_variables: The design variables to extract bounds from.

    Returns:
        A mapping from each continuous/integer variable's
        :attr:`~femtoolkit.optimization.variables.DesignVariable.path` to its
        ``(lower_bound, upper_bound)`` pair.
    """
    bounds: dict[str, tuple[float, float]] = {}
    for variable in design_variables:
        if variable.lower_bound is not None and variable.upper_bound is not None:
            bounds[variable.path] = (float(variable.lower_bound), float(variable.upper_bound))
    return bounds


def normalized_distance(
    point_a: dict[str, float], point_b: dict[str, float], bounds: dict[str, tuple[float, float]]
) -> float:
    """The normalized Euclidean distance between two points in design space.

    Each dimension is first rescaled to ``z_i = (x_i - x_min) / (x_max - x_min)``
    before computing distance, so a variable with a large physical range (e.g. a load
    in Newtons) cannot dominate a variable with a small one (e.g. a thickness in
    meters) purely because of unit choice (spec section 9). Only dimensions present in
    both points and in ``bounds`` are compared.

    Args:
        point_a: The first point, keyed by design-variable path.
        point_b: The second point, keyed by design-variable path.
        bounds: Each dimension's ``(lower, upper)`` bound (see
            :func:`design_variable_bounds`).

    Returns:
        The normalized Euclidean distance, in ``[0, sqrt(len(bounds))]``. ``0.0`` if no
        comparable dimension exists.
    """
    squared_sum = 0.0
    for name, (lower, upper) in bounds.items():
        if name not in point_a or name not in point_b:
            continue
        width = upper - lower
        if width <= 0.0:
            # A zero-width range contributes nothing -- every value along that
            # dimension is identical, so it carries no discriminating information.
            continue
        z_a = (point_a[name] - lower) / width
        z_b = (point_b[name] - lower) / width
        squared_sum += (z_a - z_b) ** 2
    return math.sqrt(squared_sum)


def exploration_score(
    point: dict[str, float], dataset: SnapshotDataset, bounds: dict[str, tuple[float, float]]
) -> float:
    """How far ``point`` lies from every existing training sample, in normalized space.

    Args:
        point: The candidate point, keyed by design-variable path.
        dataset: The current training dataset.
        bounds: Each dimension's ``(lower, upper)`` bound.

    Returns:
        The minimum normalized distance from ``point`` to any snapshot in
        ``dataset`` -- larger means more poorly covered, and therefore a higher
        exploration priority. Returns ``0.0`` if ``dataset`` has no snapshots.
    """
    if not dataset.snapshots:
        return 0.0
    return min(
        normalized_distance(point, snapshot.inputs, bounds) for snapshot in dataset.snapshots
    )


def error_score(domain_status: DomainStatus) -> float:
    """How much a candidate's applicability-domain status counts as an error-based priority.

    Args:
        domain_status: The candidate's :class:`~femtoolkit.surrogate.domain.DomainStatus`,
            from a surrogate prediction at that point.

    Returns:
        ``1.0`` (highest priority) for a point outside the training domain or
        invalid, ``0.5`` for a boundary point, ``0.0`` for a point safely within the
        training domain.
    """
    return _DOMAIN_ERROR_PROXY[domain_status]


@dataclass(frozen=True)
class CandidateScore:
    """One scored candidate from :func:`rank_candidates`.

    Attributes:
        point: The candidate's design-variable values, keyed by variable name
            (matching
            :class:`~femtoolkit.optimization.evaluation.DesignEvaluation.design_variables`).
        evaluation: The candidate's surrogate-derived
            :class:`~femtoolkit.optimization.evaluation.DesignEvaluation`.
        exploration: The normalized exploration score, in ``[0, sqrt(n_dims)]``.
        objective: The normalized objective (exploitation) score, in ``[0, 1]``
            (``1.0`` is the most promising candidate in the scored pool).
        error: The error-based priority score (see :func:`error_score`), in ``[0, 1]``.
        score: The final combined score for the strategy that was used -- higher is a
            higher priority for the next high-fidelity evaluation.
    """

    point: dict[str, float]
    evaluation: DesignEvaluation
    exploration: float
    objective: float
    error: float
    score: float


def rank_candidates(
    candidates: list[dict[str, float]],
    dataset: SnapshotDataset,
    design_variables: list[DesignVariable],
    evaluator: SurrogateEvaluator,
    objective: Objective,
    strategy: SamplingStrategy = SamplingStrategy.HYBRID,
    exploration_weight: float = 0.5,
    exploitation_weight: float = 0.5,
) -> list[CandidateScore]:
    """Score and rank a candidate pool by the configured adaptive-sampling strategy.

    Every candidate is first evaluated by the surrogate (never by the high-fidelity
    model -- see :mod:`femtoolkit.surrogate.workflows.evaluator`'s module docstring);
    a candidate whose design-variable values are invalid, or that the surrogate cannot
    predict, is dropped rather than scored.

    Args:
        candidates: The candidate pool, each keyed by design-variable *name* (see
            :func:`~femtoolkit.adaptive.candidates.generate_candidate_pool`).
        dataset: The current training dataset (used for the exploration score).
        design_variables: The design variables the candidates were drawn from.
        evaluator: A :class:`~femtoolkit.surrogate.workflows.evaluator.SurrogateEvaluator`
            wrapping the current surrogate(s).
        objective: The objective driving the exploitation score and feasibility-first
            ranking.
        strategy: Which :class:`SamplingStrategy` to combine scores with.
        exploration_weight: The hybrid strategy's :math:`w_e`.
        exploitation_weight: The hybrid strategy's :math:`w_o`.

    Returns:
        One :class:`CandidateScore` per successfully evaluated candidate, sorted by
        ``score`` descending (highest priority first).
    """
    bounds = design_variable_bounds(design_variables)
    path_by_name = {variable.name: variable.path for variable in design_variables}

    evaluations: list[DesignEvaluation] = []
    exploration_values: list[float] = []
    error_values: list[float] = []
    points: list[dict[str, float]] = []

    for point in candidates:
        evaluation = evaluator.evaluate(
            f"candidate-{len(points)}", point, design_variables, [objective], []
        )
        if evaluation.status in (DesignStatus.FAILED, DesignStatus.INVALID):
            continue
        path_point = {
            path_by_name[name]: value for name, value in point.items() if name in path_by_name
        }
        info = evaluation.metadata.get("surrogate_info", {}).get(objective.name, {})
        domain_status_value = info.get("domain_status", DomainStatus.WITHIN_TRAINING_DOMAIN.value)
        domain_status = DomainStatus(domain_status_value)

        points.append(point)
        evaluations.append(evaluation)
        exploration_values.append(exploration_score(path_point, dataset, bounds))
        error_values.append(error_score(domain_status))

    if not evaluations:
        return []

    objective_values = _normalized_objective_scores(evaluations, objective)
    max_exploration = max(exploration_values) or 1.0

    scored: list[CandidateScore] = []
    for point, evaluation, exploration, objective_score, error in zip(
        points, evaluations, exploration_values, objective_values, error_values, strict=True
    ):
        normalized_exploration = exploration / max_exploration
        if strategy is SamplingStrategy.DISTANCE:
            combined = normalized_exploration
        elif strategy is SamplingStrategy.ERROR:
            combined = error
        elif strategy is SamplingStrategy.OBJECTIVE:
            combined = objective_score
        else:
            combined = (
                exploration_weight * normalized_exploration + exploitation_weight * objective_score
            )
        scored.append(
            CandidateScore(
                point=point,
                evaluation=evaluation,
                exploration=exploration,
                objective=objective_score,
                error=error,
                score=combined,
            )
        )

    scored.sort(key=lambda candidate: candidate.score, reverse=True)
    return scored


def _normalized_objective_scores(
    evaluations: list[DesignEvaluation], objective: Objective
) -> list[float]:
    """Rank-based objective score in ``[0, 1]`` -- ``1.0`` is the most promising design.

    Uses the same feasibility-first comparison every optimization algorithm in this
    toolkit already uses (:func:`~femtoolkit.optimization.evaluation.is_better_evaluation`)
    rather than a numeric min-max normalization, so an infeasible design never
    outranks a feasible one regardless of how good its raw objective value looks.
    """
    order = list(range(len(evaluations)))

    def _worse(i: int, j: int) -> bool:
        return is_better_evaluation(evaluations[j], evaluations[i], objective)

    # A simple, transparent insertion sort by pairwise comparison -- the candidate
    # pools this module scores are small (tens, not millions, of points), so O(n^2)
    # is not worth trading away the readability of reusing `is_better_evaluation`
    # directly as the sort key.
    for i in range(1, len(order)):
        key = order[i]
        j = i - 1
        while j >= 0 and _worse(order[j], key):
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = key

    n = len(order)
    scores = [0.0] * n
    if n == 1:
        scores[order[0]] = 1.0
    else:
        for rank, index in enumerate(order):
            scores[index] = 1.0 - rank / (n - 1)
    return scores


__all__ = [
    "CandidateScore",
    "SamplingStrategy",
    "design_variable_bounds",
    "error_score",
    "exploration_score",
    "normalized_distance",
    "rank_candidates",
]
