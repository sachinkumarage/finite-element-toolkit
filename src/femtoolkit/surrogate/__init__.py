"""Reduced-order modeling and surrogate-based engineering analysis (Version 35).

**The high-fidelity model.** Every analysis so far in this toolkit
computes

.. code-block:: text

    y = f(x)

where ``x`` is a vector of input/design variables (Young's modulus,
density, thickness, load, temperature, ...) and ``f`` is the expensive
FEA simulation itself -- an assembled and solved
:class:`~femtoolkit.application.project.Project`. ``y`` is one or more
scalar engineering responses: maximum displacement, maximum stress, a
reaction force, a natural frequency, a temperature, a heat flux, a
mass. This package never replaces ``f`` -- it only ever builds fast,
transparent approximations of it from snapshots of real FEA results,
and keeps every approximation clearly labeled as such.

**Reduced-order modeling.** Instead of re-solving the full model for
every new input, a reduced-order model approximates its (potentially
very large) full-field solution with a much smaller representation:

.. code-block:: text

    u ~ V q

where ``u`` is the full FEA solution (e.g. every nodal displacement
DOF), ``V`` is a reduced basis, and ``q`` is a small vector of reduced
coordinates. :mod:`femtoolkit.surrogate.rom` builds ``V`` via Proper
Orthogonal Decomposition (POD, an SVD of a snapshot matrix of several
full-field solutions).

.. code-block:: text

    High-Fidelity FEA
            |
            v
    Snapshot Data
            |
            v
    Reduced Basis
            |
            v
    Reduced Model
            |
            v
    Fast Prediction

**Surrogate modeling.** For a single *scalar* response, a surrogate
model (:mod:`femtoolkit.surrogate.models`) fits a transparent,
interpretable function directly from design variables to that response
-- a low-order polynomial regression, or a radial basis function
interpolant -- instead of reducing a full field.

.. code-block:: text

    Parameter Space
           |
           v
    Version 34 parallel FEA
           |
           v
    Snapshot Dataset
           |
           v
    Train Surrogate / ROM
           |
           v
    Validation
           |
           v
    Fast Prediction

**The high-fidelity FEA model remains the source of truth.** A
surrogate or ROM prediction is always explicitly labeled as such (see
:class:`~femtoolkit.surrogate.models.base.SurrogatePrediction`), always
carries an applicability-domain status
(:mod:`femtoolkit.surrogate.domain`), and is never automatically
accepted as a final engineering result -- see
:mod:`femtoolkit.surrogate.workflows.verification` for checking a
surrogate prediction (or a surrogate-optimized design) against real
FEA before any such acceptance.

**Explicit scope exclusions (this version).** No neural networks, no
deep learning, no Gaussian process regression, no Bayesian
optimization, no advanced active learning, no topology/shape
optimization, no adjoint methods, no distributed/GPU surrogate
training, and no formal uncertainty quantification of a surrogate's own
prediction error -- see ``docs/releases/v35.0.0.md`` for the full scope
boundary and the Version 36 preview.
"""

from __future__ import annotations

from femtoolkit.surrogate.datasets import (
    DatasetSplit,
    Snapshot,
    SnapshotDataset,
    collect_snapshots_from_runs,
)
from femtoolkit.surrogate.domain import (
    ApplicabilityDomain,
    DomainCheckResult,
    DomainStatus,
    FeatureBounds,
)
from femtoolkit.surrogate.metrics import MetricSet, compute_metrics
from femtoolkit.surrogate.models import (
    SURROGATE_MODEL_TYPES,
    PolynomialRegressionSurrogate,
    PredictionStatus,
    RBFSurrogate,
    SurrogateModel,
    SurrogatePrediction,
    build_surrogate_model,
)
from femtoolkit.surrogate.persistence import (
    load_rom_model,
    load_surrogate_model,
    save_rom_model,
    save_surrogate_model,
)
from femtoolkit.surrogate.rom import (
    FieldSnapshot,
    PODModel,
    ReconstructionError,
    RomValidationReport,
)
from femtoolkit.surrogate.scaling import (
    SCALER_TYPES,
    MinMaxScaler,
    Scaler,
    StandardScaler,
    build_scaler,
)
from femtoolkit.surrogate.validation import (
    CrossValidationResult,
    EngineeringTolerance,
    SurrogateValidationReport,
    k_fold_cross_validate,
)
from femtoolkit.surrogate.workflows import (
    AcceptanceStatus,
    EvaluationBackend,
    HighFidelityEvaluator,
    SurrogateEvaluator,
    SurrogateVerificationRecord,
    TrainingConfig,
    generate_training_dataset,
    recommend_candidates,
    train_surrogate,
    verify_against_high_fidelity,
)

__all__ = [
    "SCALER_TYPES",
    "SURROGATE_MODEL_TYPES",
    "AcceptanceStatus",
    "ApplicabilityDomain",
    "CrossValidationResult",
    "DatasetSplit",
    "DomainCheckResult",
    "DomainStatus",
    "EngineeringTolerance",
    "EvaluationBackend",
    "FeatureBounds",
    "FieldSnapshot",
    "HighFidelityEvaluator",
    "MetricSet",
    "MinMaxScaler",
    "PODModel",
    "PolynomialRegressionSurrogate",
    "PredictionStatus",
    "RBFSurrogate",
    "ReconstructionError",
    "RomValidationReport",
    "Scaler",
    "Snapshot",
    "SnapshotDataset",
    "StandardScaler",
    "SurrogateEvaluator",
    "SurrogateModel",
    "SurrogatePrediction",
    "SurrogateValidationReport",
    "SurrogateVerificationRecord",
    "TrainingConfig",
    "build_scaler",
    "build_surrogate_model",
    "collect_snapshots_from_runs",
    "compute_metrics",
    "generate_training_dataset",
    "k_fold_cross_validate",
    "load_rom_model",
    "load_surrogate_model",
    "recommend_candidates",
    "save_rom_model",
    "save_surrogate_model",
    "train_surrogate",
    "verify_against_high_fidelity",
]
