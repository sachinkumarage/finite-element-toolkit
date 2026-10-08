"""The structured outcome of an adaptive, surrogate-assisted engineering study (Version 36)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from femtoolkit.adaptive.refinement import RefinementStepResult, SurrogateAcceptanceState
from femtoolkit.surrogate.validation import SurrogateValidationReport


@dataclass
class AdaptiveStudyResult:
    """The complete, structured result of running an
    :class:`~femtoolkit.adaptive.study.AdaptiveStudy`.

    Attributes:
        initial_sample_count: How many high-fidelity snapshots the study started with.
        total_high_fidelity_evaluations: Every high-fidelity evaluation performed by
            this study, initial samples included.
        surrogate_evaluations: Every surrogate (candidate-pool) evaluation performed.
        iteration_count: How many adaptive-refinement iterations actually ran.
        best_verified_design: The design point, keyed by path, with the best
            high-fidelity-verified objective value seen -- ``None`` if no
            high-fidelity evaluation ever completed.
        best_verified_objective: The objective value at ``best_verified_design``,
            from the high-fidelity result itself (never the surrogate's prediction).
        best_surrogate_predicted_design: The design point the surrogate currently
            believes is best, keyed by design-variable *name* (matching
            :attr:`~femtoolkit.optimization.evaluation.DesignEvaluation.design_variables`;
            contrast with ``best_verified_design``, keyed by override *path*), from
            the most recent candidate search -- ``None`` if no candidate search ever
            completed.
        best_surrogate_predicted_objective: The surrogate's predicted objective value
            at ``best_surrogate_predicted_design``.
        prediction_error: The absolute/relative error between the surrogate's
            prediction and the high-fidelity result at ``best_verified_design``, keyed
            by response name (empty if unavailable).
        convergence_history: The best-verified objective value after each iteration,
            in iteration order.
        iteration_history: Every :class:`~femtoolkit.adaptive.refinement.RefinementStepResult`,
            in iteration order.
        final_surrogate_metrics: The surrogate's validation report after the last
            successful retraining, or ``None``.
        status: The overall engineering status of ``best_verified_design``.
        stopping_reason: A short, human-readable description of why the study stopped.

    A design here is never reported as "optimal": only as the best *verified* design
    found within the study's configured budget (spec section 16).
    """

    initial_sample_count: int
    total_high_fidelity_evaluations: int
    surrogate_evaluations: int
    iteration_count: int
    best_verified_design: dict[str, float] | None
    best_verified_objective: float | None
    best_surrogate_predicted_design: dict[str, float] | None
    best_surrogate_predicted_objective: float | None
    prediction_error: dict[str, float] = field(default_factory=dict)
    convergence_history: list[float | None] = field(default_factory=list)
    iteration_history: list[RefinementStepResult] = field(default_factory=list)
    final_surrogate_metrics: SurrogateValidationReport | None = None
    status: SurrogateAcceptanceState = SurrogateAcceptanceState.SURROGATE_ONLY
    stopping_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable summary (iteration history omitted; large/complex)."""
        return {
            "initial_sample_count": self.initial_sample_count,
            "total_high_fidelity_evaluations": self.total_high_fidelity_evaluations,
            "surrogate_evaluations": self.surrogate_evaluations,
            "iteration_count": self.iteration_count,
            "best_verified_design": self.best_verified_design,
            "best_verified_objective": self.best_verified_objective,
            "best_surrogate_predicted_design": self.best_surrogate_predicted_design,
            "best_surrogate_predicted_objective": self.best_surrogate_predicted_objective,
            "prediction_error": dict(self.prediction_error),
            "convergence_history": list(self.convergence_history),
            "status": self.status.value,
            "stopping_reason": self.stopping_reason,
        }


__all__ = ["AdaptiveStudyResult"]
