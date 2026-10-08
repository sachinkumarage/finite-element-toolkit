"""Adaptive-sampling candidate recommendation (Version 35, foundation only).

.. code-block:: text

    Surrogate
        |
        v
    Identify candidate
        |
        v
    Recommend FEA evaluation
        |
        v
    User / study approves
        |
        v
    High-fidelity FEA
        |
        v
    Add snapshot
        |
        v
    Retrain

This is deliberately a *foundation*, not an active-learning algorithm:
:func:`recommend_candidates` only ever scores and ranks a caller-supplied
pool of candidate points -- it never generates new candidate points
itself, and it never triggers a high-fidelity FEA run on its own. A new
evaluation always requires an explicit, separate call into the
existing simulation infrastructure, approved by the calling code or a
human (see the module docstring's final two steps above).
"""

from __future__ import annotations

from dataclasses import dataclass

from femtoolkit.surrogate.domain import DomainStatus
from femtoolkit.surrogate.models.base import PredictionStatus, SurrogateModel

_DOMAIN_PRIORITY = {
    DomainStatus.OUTSIDE_TRAINING_DOMAIN: 2.0,
    DomainStatus.BOUNDARY: 1.0,
    DomainStatus.WITHIN_TRAINING_DOMAIN: 0.0,
}


@dataclass(frozen=True)
class SamplingCandidate:
    """One candidate point recommended for a future high-fidelity FEA evaluation.

    Attributes:
        point: The candidate's input/feature values.
        domain_status: Why this point was flagged (its applicability-domain status).
        reason: A short, human-readable explanation.
        priority: A relative ranking score -- higher means more worth evaluating next.
    """

    point: dict[str, float]
    domain_status: DomainStatus
    reason: str
    priority: float


def recommend_candidates(
    model: SurrogateModel,
    candidate_points: list[dict[str, float]],
    max_recommendations: int = 5,
) -> list[SamplingCandidate]:
    """Rank a pool of candidate points by how poorly covered they are by the training data.

    Args:
        model: The fitted surrogate model whose training domain defines "coverage".
        candidate_points: The candidate pool to score -- e.g. a
            denser grid over the design space, or points sampled from
            an uncertainty study.
        max_recommendations: The maximum number of candidates to return.

    Returns:
        Up to ``max_recommendations`` :class:`SamplingCandidate` objects
        whose domain status is not
        :attr:`~femtoolkit.surrogate.domain.DomainStatus.WITHIN_TRAINING_DOMAIN`,
        most poorly covered first. Empty if every candidate is already
        well within the training domain, or if ``candidate_points`` is empty.
    """
    scored: list[SamplingCandidate] = []
    for point in candidate_points:
        prediction = model.predict_point(point)
        if prediction.status is PredictionStatus.INVALID_INPUT:
            continue
        priority = _DOMAIN_PRIORITY.get(prediction.domain_status, 0.0)
        if priority <= 0.0:
            continue
        scored.append(
            SamplingCandidate(
                point=dict(point),
                domain_status=prediction.domain_status,
                reason=(
                    f"Candidate lies {prediction.domain_status.value} of the current "
                    f"training data ({model.training_metadata.dataset_id!r}, "
                    f"v{model.training_metadata.dataset_version})."
                ),
                priority=priority,
            )
        )
    scored.sort(key=lambda candidate: candidate.priority, reverse=True)
    return scored[:max_recommendations]


__all__ = ["SamplingCandidate", "recommend_candidates"]
