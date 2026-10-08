"""Surrogate model implementations (Version 35).

.. code-block:: text

    SurrogateModel (base.py)
    |-- PolynomialRegressionSurrogate (polynomial.py)
    `-- RBFSurrogate (rbf.py)

Deliberately a small, transparent set -- see ``femtoolkit.surrogate``'s
package docstring for why black-box machine learning is out of scope.
"""

from __future__ import annotations

from femtoolkit.surrogate.models.base import (
    PredictionStatus,
    SurrogateModel,
    SurrogatePrediction,
    TrainingMetadata,
)
from femtoolkit.surrogate.models.polynomial import SUPPORTED_DEGREES, PolynomialRegressionSurrogate
from femtoolkit.surrogate.models.rbf import DEFAULT_REGULARIZATION, SUPPORTED_KERNELS, RBFSurrogate

SURROGATE_MODEL_TYPES = ("polynomial", "rbf")
"""The surrogate model type names :func:`build_surrogate_model` accepts."""


def build_surrogate_model(model_type: str, **kwargs: object) -> SurrogateModel:
    """Construct a fresh, unfitted surrogate model by type name.

    Args:
        model_type: One of :data:`SURROGATE_MODEL_TYPES`.
        **kwargs: Forwarded to the concrete model's constructor.

    Returns:
        A new, unfitted :class:`SurrogateModel`.

    Raises:
        ValueError: If ``model_type`` is unknown.
    """
    if model_type == "polynomial":
        return PolynomialRegressionSurrogate(**kwargs)
    if model_type == "rbf":
        return RBFSurrogate(**kwargs)
    raise ValueError(f"Unknown model_type {model_type!r}; expected one of {SURROGATE_MODEL_TYPES}.")


__all__ = [
    "DEFAULT_REGULARIZATION",
    "SUPPORTED_DEGREES",
    "SUPPORTED_KERNELS",
    "SURROGATE_MODEL_TYPES",
    "PolynomialRegressionSurrogate",
    "PredictionStatus",
    "RBFSurrogate",
    "SurrogateModel",
    "SurrogatePrediction",
    "TrainingMetadata",
    "build_surrogate_model",
]
