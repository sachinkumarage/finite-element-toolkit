"""Multi-fidelity engineering modeling (Version 37).

**Engineering concept.** The same physical quantity can be estimated by models
of very different accuracy, physical detail, and computational cost:

.. code-block:: text

    Low Fidelity   -> fast but less accurate
    High Fidelity  -> slower but more accurate

Let :math:`y_L(x)` be the low-fidelity result and :math:`y_H(x)` the
high-fidelity result for design point :math:`x`. The **model discrepancy** is

.. math::

    \\delta(x) = y_H(x) - y_L(x)

and the simplest fused model -- the main method this version implements -- adds
a surrogate of that discrepancy back onto the cheap low-fidelity result:

.. math::

    \\hat y_H(x) = y_L(x) + \\hat\\delta(x)

.. code-block:: text

    Design Point
         |
         v
    Low-Fidelity Model
         |
         v
    Low-Fidelity Result
         |
         v
    Discrepancy Model          (femtoolkit.multifidelity.discrepancy -- reuses
         |                       the unmodified Version 35 surrogate machinery)
         v
    Fused Prediction
         |
         v
    Optional High-Fidelity FEA
         |
         v
    Validation

**The fused prediction is never the same thing as a verified high-fidelity FEA
result.** Every :class:`~femtoolkit.multifidelity.model.FusedPrediction` keeps
:math:`y_L`, :math:`\\hat\\delta`, and :math:`\\hat y_H` as three separate
fields, and :func:`~femtoolkit.multifidelity.model.verify_fused_prediction`
checks the fused prediction against a freshly-run real high-fidelity result
before any such claim can be made -- it never assumes the correction helps; see
:mod:`femtoolkit.multifidelity.validation`.

Built entirely on top of the existing Version 30 simulation workflows
(:mod:`femtoolkit.studies`, :mod:`femtoolkit.runs`) and Version 35 surrogate
machinery (:mod:`femtoolkit.surrogate`) -- no new regression, dataset, or
metric logic is duplicated here.

**Explicit scope exclusions (this version).** No Gaussian processes,
co-kriging, deep learning, neural networks, Bayesian optimization,
topology/shape optimization, digital twins, distributed/GPU/cloud computing,
advanced reliability methods, or adaptive fidelity selection -- see
``docs/releases/v37.0.0.md`` for the full scope boundary and the Version 38
preview.
"""

from __future__ import annotations

from femtoolkit.multifidelity.dataset import (
    MultiFidelityDataset,
    MultiFidelitySample,
    compute_discrepancy,
)
from femtoolkit.multifidelity.discrepancy import train_discrepancy_surrogate
from femtoolkit.multifidelity.fidelity import (
    HIGH_FIDELITY,
    LOW_FIDELITY,
    AnalyticalFidelityModel,
    FidelityLevel,
    FidelityModel,
    SimulationFidelityModel,
    summarize_costs,
)
from femtoolkit.multifidelity.model import (
    FusedPrediction,
    FusionAcceptanceStatus,
    FusionVerificationRecord,
    MultiFidelityModel,
    verify_fused_prediction,
)
from femtoolkit.multifidelity.validation import FidelityComparisonReport, compare_fidelity_accuracy

__all__ = [
    "HIGH_FIDELITY",
    "LOW_FIDELITY",
    "AnalyticalFidelityModel",
    "FidelityComparisonReport",
    "FidelityLevel",
    "FidelityModel",
    "FusedPrediction",
    "FusionAcceptanceStatus",
    "FusionVerificationRecord",
    "MultiFidelityDataset",
    "MultiFidelityModel",
    "MultiFidelitySample",
    "SimulationFidelityModel",
    "compare_fidelity_accuracy",
    "compute_discrepancy",
    "summarize_costs",
    "train_discrepancy_surrogate",
    "verify_fused_prediction",
]
