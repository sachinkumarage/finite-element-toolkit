"""Radial Basis Function (RBF) surrogate (Version 35).

.. math::

    \\hat f(x) = \\sum_{i=1}^{N} w_i \\, \\phi(\\|x - x_i\\|)

Each training point ``x_i`` becomes one radial basis function centered
at that point; the weights ``w_i`` are solved so the interpolant passes
through (or, with regularization, closely fits) every training
response. Two standard, numerically simple kernels are supported:

- **Gaussian:** :math:`\\phi(r) = \\exp(-(r/\\varepsilon)^2)`
- **Multiquadric:** :math:`\\phi(r) = \\sqrt{r^2 + \\varepsilon^2}`

**Numerical stability.** The interpolation matrix
:math:`\\Phi_{ij} = \\phi(\\|x_i - x_j\\|)` can become ill-conditioned
when training points are close together or the shape parameter
``epsilon`` is poorly chosen. This implementation (a) scales inputs
before computing distances (see
:class:`~femtoolkit.surrogate.models.base.SurrogateModel`), which keeps
every feature's contribution to the distance comparable, and (b) adds a
small Tikhonov regularization term (``regularization``) to the
interpolation matrix's diagonal before solving, reusing the same
:func:`numpy.linalg.lstsq` safety net
:mod:`femtoolkit.surrogate.models.polynomial` uses rather than a direct
(potentially singular) matrix inverse.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from femtoolkit.exceptions import SurrogateFittingError, ValidationError
from femtoolkit.surrogate.models.base import SurrogateModel

SUPPORTED_KERNELS = ("gaussian", "multiquadric")
"""The supported radial basis function kernels -- see the module docstring."""

DEFAULT_REGULARIZATION = 1e-8
"""The default Tikhonov regularization added to the interpolation matrix's diagonal."""


def _kernel(distances: np.ndarray, kernel: str, epsilon: float) -> np.ndarray:
    if kernel == "gaussian":
        return np.exp(-((distances / epsilon) ** 2))
    if kernel == "multiquadric":
        return np.sqrt(distances**2 + epsilon**2)
    raise ValidationError(f"Unknown RBF kernel {kernel!r}; expected one of {SUPPORTED_KERNELS}.")


class RBFSurrogate(SurrogateModel):
    """A radial basis function interpolation surrogate.

    Attributes:
        kernel: ``"gaussian"`` or ``"multiquadric"``.
        epsilon: The kernel's shape parameter. ``None`` (the default)
            picks a data-driven value at fit time: the mean pairwise
            distance between (scaled) training points, a simple,
            transparent heuristic that keeps the kernel's width
            comparable to the training data's own spread.
        regularization: The Tikhonov regularization added to the
            interpolation matrix's diagonal before solving.
    """

    name = "rbf"

    def __init__(
        self,
        kernel: str = "gaussian",
        epsilon: float | None = None,
        regularization: float = DEFAULT_REGULARIZATION,
        feature_scaler_type: str = "standard",
        response_scaler_type: str = "standard",
    ) -> None:
        """Create an unfitted RBF surrogate.

        Args:
            kernel: See :attr:`kernel`.
            epsilon: See :attr:`epsilon`.
            regularization: See :attr:`regularization`.
            feature_scaler_type: See :class:`~femtoolkit.surrogate.models.base.SurrogateModel`.
            response_scaler_type: See :class:`~femtoolkit.surrogate.models.base.SurrogateModel`.

        Raises:
            SurrogateFittingError: If ``kernel`` is not one of :data:`SUPPORTED_KERNELS`,
                or ``regularization`` is negative.
        """
        super().__init__(feature_scaler_type, response_scaler_type)
        if kernel not in SUPPORTED_KERNELS:
            raise SurrogateFittingError(
                f"Unsupported RBF kernel {kernel!r}; expected one of {SUPPORTED_KERNELS}."
            )
        if regularization < 0.0:
            raise SurrogateFittingError(
                f"regularization must be non-negative, got {regularization}."
            )
        self.kernel = kernel
        self.epsilon = epsilon
        self.regularization = regularization
        self._centers: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._fitted_epsilon: float | None = None

    def _fit_scaled(self, x_scaled: np.ndarray, y_scaled: np.ndarray) -> None:
        n_samples = x_scaled.shape[0]
        if n_samples < 2:
            raise SurrogateFittingError("RBF surrogate requires at least 2 training points.")
        distances = cdist(x_scaled, x_scaled)

        epsilon = self.epsilon
        if epsilon is None:
            off_diagonal = distances[~np.eye(n_samples, dtype=bool)]
            epsilon = float(off_diagonal.mean()) if off_diagonal.size else 1.0
            if epsilon < 1e-12:
                epsilon = 1.0

        phi = _kernel(distances, self.kernel, epsilon)
        phi_regularized = phi + self.regularization * np.eye(n_samples)
        weights, _residuals, _rank, _singular_values = np.linalg.lstsq(
            phi_regularized, y_scaled, rcond=None
        )
        if not np.isfinite(weights).all():
            raise SurrogateFittingError(
                "RBF surrogate produced non-finite weights -- the interpolation matrix is "
                "likely severely ill-conditioned; try increasing regularization or epsilon."
            )
        self._centers = x_scaled
        self._weights = weights
        self._fitted_epsilon = epsilon

    def _predict_scaled(self, x_scaled: np.ndarray) -> np.ndarray:
        distances = cdist(x_scaled, self._centers)
        phi = _kernel(distances, self.kernel, self._fitted_epsilon)
        return phi @ self._weights

    def model_parameters(self) -> dict[str, Any]:
        return {
            "kernel": self.kernel,
            "epsilon": self.epsilon,
            "fitted_epsilon": self._fitted_epsilon,
            "regularization": self.regularization,
            "centers": self._centers.tolist(),
            "weights": self._weights.tolist(),
        }

    def load_model_parameters(self, parameters: dict[str, Any]) -> None:
        self.kernel = parameters["kernel"]
        self.epsilon = parameters["epsilon"]
        self.regularization = parameters["regularization"]
        self._fitted_epsilon = parameters["fitted_epsilon"]
        self._centers = np.asarray(parameters["centers"], dtype=float)
        self._weights = np.asarray(parameters["weights"], dtype=float)


__all__ = ["DEFAULT_REGULARIZATION", "SUPPORTED_KERNELS", "RBFSurrogate"]
