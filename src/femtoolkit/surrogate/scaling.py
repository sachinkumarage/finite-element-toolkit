"""Feature and response scaling for surrogate models (Version 35).

**Why scaling matters.** Engineering variables span wildly different
numerical magnitudes -- a Young's modulus (~1e11 Pa), a density
(~1e3 kg/m^3), a plate thickness (~1e-2 m), a load (~1e3 N), and a
temperature (~1e2 K) might all be inputs to the same surrogate. Fitting
a model directly on these raw magnitudes lets the largest-magnitude
variable dominate a least-squares fit or a distance-based kernel (RBF)
for reasons that have nothing to do with its actual engineering
importance. Scaling removes this artifact by transforming every
variable onto a comparable numerical range before fitting, and
transforming back afterward.

Every :class:`Scaler` is fit using *training data only* -- never on
validation or test data, and never on the full dataset before
splitting -- so a reported validation/test score is never contaminated
by information leaked from the data it is supposed to be unseen on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from femtoolkit.exceptions import InvalidScalingConfigurationError


class Scaler(ABC):
    """A reusable, invertible column-wise transformation fit on training data only."""

    _fitted: bool = False

    @abstractmethod
    def fit(self, data: np.ndarray) -> Scaler:
        """Fit this scaler's parameters from ``data`` (shape ``(n_samples, n_columns)``)."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply the fitted transformation to ``data``."""
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Undo the fitted transformation, returning values in the original (physical) units."""
        raise NotImplementedError

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit this scaler on ``data``, then immediately transform it."""
        return self.fit(data).transform(data)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise InvalidScalingConfigurationError(
                f"{type(self).__name__} must be fit() before it can transform data."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable representation of this fitted scaler."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scaler:
        """Reconstruct a fitted scaler from :meth:`to_dict`'s output."""
        kind = data["type"]
        if kind == "standard":
            return StandardScaler._from_state(data)
        if kind == "minmax":
            return MinMaxScaler._from_state(data)
        if kind == "identity":
            return IdentityScaler()
        raise InvalidScalingConfigurationError(f"Unknown scaler type {kind!r}.")


@dataclass
class StandardScaler(Scaler):
    """Standardization: ``z = (x - mean) / std``, per column.

    A zero-variance column (a constant input/output across every
    training snapshot) is handled explicitly: its scale is treated as
    ``1.0`` instead of dividing by zero, so a constant column simply
    becomes its own (mean-centered) value rather than producing ``NaN``.
    """

    mean_: np.ndarray = field(default=None, repr=False)
    scale_: np.ndarray = field(default=None, repr=False)

    def fit(self, data: np.ndarray) -> StandardScaler:
        data = np.asarray(data, dtype=float)
        if data.size == 0:
            raise InvalidScalingConfigurationError("Cannot fit a scaler on empty data.")
        self.mean_ = data.mean(axis=0)
        std = data.std(axis=0)
        self.scale_ = np.where(std < 1e-12, 1.0, std)
        self._fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return (np.asarray(data, dtype=float) - self.mean_) / self.scale_

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return np.asarray(data, dtype=float) * self.scale_ + self.mean_

    def to_dict(self) -> dict[str, Any]:
        self._require_fitted()
        return {"type": "standard", "mean": self.mean_.tolist(), "scale": self.scale_.tolist()}

    @classmethod
    def _from_state(cls, data: dict[str, Any]) -> StandardScaler:
        scaler = cls(
            mean_=np.asarray(data["mean"], dtype=float),
            scale_=np.asarray(data["scale"], dtype=float),
        )
        scaler._fitted = True
        return scaler


@dataclass
class MinMaxScaler(Scaler):
    """Min-max scaling: ``z = (x - min) / (max - min)``, per column, onto ``[0, 1]``.

    A zero-range column is handled the same way as
    :class:`StandardScaler`'s zero-variance column: the range is
    treated as ``1.0`` rather than dividing by zero.
    """

    min_: np.ndarray = field(default=None, repr=False)
    range_: np.ndarray = field(default=None, repr=False)

    def fit(self, data: np.ndarray) -> MinMaxScaler:
        data = np.asarray(data, dtype=float)
        if data.size == 0:
            raise InvalidScalingConfigurationError("Cannot fit a scaler on empty data.")
        self.min_ = data.min(axis=0)
        span = data.max(axis=0) - self.min_
        self.range_ = np.where(span < 1e-12, 1.0, span)
        self._fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return (np.asarray(data, dtype=float) - self.min_) / self.range_

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return np.asarray(data, dtype=float) * self.range_ + self.min_

    def to_dict(self) -> dict[str, Any]:
        self._require_fitted()
        return {"type": "minmax", "min": self.min_.tolist(), "range": self.range_.tolist()}

    @classmethod
    def _from_state(cls, data: dict[str, Any]) -> MinMaxScaler:
        scaler = cls(
            min_=np.asarray(data["min"], dtype=float),
            range_=np.asarray(data["range"], dtype=float),
        )
        scaler._fitted = True
        return scaler


class IdentityScaler(Scaler):
    """A no-op scaler (returns its input unchanged), used when scaling is explicitly disabled."""

    def __init__(self) -> None:
        self._fitted = True

    def fit(self, data: np.ndarray) -> IdentityScaler:
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(data, dtype=float)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(data, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "identity"}


SCALER_TYPES = ("standard", "minmax", "identity")
"""The supported scaler type names, used by :func:`build_scaler` and persistence."""


def build_scaler(scaler_type: str) -> Scaler:
    """Construct a fresh, unfitted :class:`Scaler` by name.

    Args:
        scaler_type: One of :data:`SCALER_TYPES`.

    Returns:
        A new, unfitted scaler.

    Raises:
        InvalidScalingConfigurationError: If ``scaler_type`` is unknown.
    """
    if scaler_type == "standard":
        return StandardScaler()
    if scaler_type == "minmax":
        return MinMaxScaler()
    if scaler_type == "identity":
        return IdentityScaler()
    raise InvalidScalingConfigurationError(
        f"Unknown scaler_type {scaler_type!r}; expected one of {SCALER_TYPES}."
    )


__all__ = [
    "SCALER_TYPES",
    "IdentityScaler",
    "MinMaxScaler",
    "Scaler",
    "StandardScaler",
    "build_scaler",
]
