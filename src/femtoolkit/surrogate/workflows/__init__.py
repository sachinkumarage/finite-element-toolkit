"""Surrogate/ROM workflows tying the framework together with the existing toolkit (Version 35)."""

from __future__ import annotations

from femtoolkit.surrogate.workflows.evaluator import (
    EvaluationBackend,
    Evaluator,
    HighFidelityEvaluator,
    SurrogateEvaluator,
    describe_uncertainty_backend,
)
from femtoolkit.surrogate.workflows.recommendation import SamplingCandidate, recommend_candidates
from femtoolkit.surrogate.workflows.training import (
    TrainingConfig,
    generate_training_dataset,
    train_surrogate,
)
from femtoolkit.surrogate.workflows.verification import (
    AcceptanceStatus,
    SurrogateVerificationRecord,
    verify_against_high_fidelity,
)

__all__ = [
    "AcceptanceStatus",
    "EvaluationBackend",
    "Evaluator",
    "HighFidelityEvaluator",
    "SamplingCandidate",
    "SurrogateEvaluator",
    "SurrogateVerificationRecord",
    "TrainingConfig",
    "describe_uncertainty_backend",
    "generate_training_dataset",
    "recommend_candidates",
    "train_surrogate",
    "verify_against_high_fidelity",
]
