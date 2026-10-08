"""Polynomial regression surrogate (Version 35).

.. math::

    y = \\beta_0 + \\sum_i \\beta_i x_i + \\sum_i \\beta_{ii} x_i^2
        + \\sum_{i<j} \\beta_{ij} x_i x_j

First-order models (``degree=1``, no quadratic/cross terms) and
second-order models (``degree=2``, the full expression above) are
supported. The regression coefficients are solved by ordinary
least-squares (:func:`numpy.linalg.lstsq`, which handles a
rank-deficient design matrix gracefully via the SVD rather than raising),
on already-scaled, physical-unit-independent inputs/outputs -- see
:class:`~femtoolkit.surrogate.models.base.SurrogateModel`.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from femtoolkit.exceptions import SurrogateFittingError
from femtoolkit.surrogate.models.base import SurrogateModel

SUPPORTED_DEGREES = (1, 2)
"""The polynomial degrees this surrogate supports -- see the module docstring."""


def _design_matrix(x_scaled: np.ndarray, degree: int) -> np.ndarray:
    n_samples, n_features = x_scaled.shape
    columns = [np.ones(n_samples)]
    columns += [x_scaled[:, i] for i in range(n_features)]
    if degree >= 2:
        columns += [x_scaled[:, i] ** 2 for i in range(n_features)]
        columns += [x_scaled[:, i] * x_scaled[:, j] for i, j in combinations(range(n_features), 2)]
    return np.column_stack(columns)


class PolynomialRegressionSurrogate(SurrogateModel):
    """A first- or second-order polynomial regression surrogate.

    Attributes:
        degree: ``1`` (linear, no interaction/quadratic terms) or
            ``2`` (full quadratic model with cross terms).
    """

    name = "polynomial"

    def __init__(
        self,
        degree: int = 2,
        feature_scaler_type: str = "standard",
        response_scaler_type: str = "standard",
    ) -> None:
        """Create an unfitted polynomial surrogate.

        Args:
            degree: See :attr:`degree`.
            feature_scaler_type: See :class:`~femtoolkit.surrogate.models.base.SurrogateModel`.
            response_scaler_type: See :class:`~femtoolkit.surrogate.models.base.SurrogateModel`.

        Raises:
            SurrogateFittingError: If ``degree`` is not one of :data:`SUPPORTED_DEGREES`.
        """
        super().__init__(feature_scaler_type, response_scaler_type)
        if degree not in SUPPORTED_DEGREES:
            raise SurrogateFittingError(
                f"Unsupported polynomial degree {degree!r}; expected one of {SUPPORTED_DEGREES}."
            )
        self.degree = degree
        self._coefficients: np.ndarray | None = None

    @property
    def n_coefficients(self) -> int | None:
        """How many regression coefficients this model has per response, once fitted."""
        return None if self._coefficients is None else self._coefficients.shape[0]

    def _fit_scaled(self, x_scaled: np.ndarray, y_scaled: np.ndarray) -> None:
        design = _design_matrix(x_scaled, self.degree)
        if design.shape[0] < design.shape[1]:
            raise SurrogateFittingError(
                f"Degree-{self.degree} polynomial surrogate needs at least {design.shape[1]} "
                f"training points (one per coefficient) but only {design.shape[0]} were given."
            )
        coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(
            design, y_scaled, rcond=None
        )
        if not np.isfinite(coefficients).all():
            raise SurrogateFittingError(
                "Polynomial regression produced non-finite coefficients -- the design matrix "
                "is likely singular or severely ill-conditioned for this data."
            )
        self._coefficients = coefficients

    def _predict_scaled(self, x_scaled: np.ndarray) -> np.ndarray:
        design = _design_matrix(x_scaled, self.degree)
        return design @ self._coefficients

    def model_parameters(self) -> dict[str, Any]:
        return {"degree": self.degree, "coefficients": self._coefficients.tolist()}

    def load_model_parameters(self, parameters: dict[str, Any]) -> None:
        self.degree = parameters["degree"]
        self._coefficients = np.asarray(parameters["coefficients"], dtype=float)


__all__ = ["SUPPORTED_DEGREES", "PolynomialRegressionSurrogate"]
