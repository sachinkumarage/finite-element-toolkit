"""Applicability domain checks for surrogate predictions (Version 35).

A surrogate model interpolates (and, dangerously, extrapolates) a
function it never actually evaluated outside the span of its training
snapshots. This module implements the minimal, transparent check this
toolkit commits to: whether a query point's features lie within,
near the edge of, or outside the training data's per-feature bounding
box. It deliberately does **not** attempt any statistical prediction-
interval or uncertainty-quantification estimate of the surrogate's own
error at a query point (out of scope for this version -- see the
package docstring).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from femtoolkit.exceptions import ValidationError

BOUNDARY_FRACTION = 0.05
"""A query point within this fraction of a feature's training range,
measured inward from either bound, is reported as `DomainStatus.BOUNDARY`
rather than squarely `WITHIN_TRAINING_DOMAIN`."""


class DomainStatus(Enum):
    """A query point's relationship to a surrogate's training domain.

    Attributes:
        WITHIN_TRAINING_DOMAIN: Every feature lies safely inside the
            training bounding box.
        BOUNDARY: Every feature lies inside the training bounding box,
            but at least one lies within :data:`BOUNDARY_FRACTION` of
            an edge.
        OUTSIDE_TRAINING_DOMAIN: At least one feature lies outside the
            training bounding box (extrapolation).
        INVALID: The query point is missing a required feature, carries
            an unexpected one, or contains a non-finite value.
    """

    WITHIN_TRAINING_DOMAIN = "within_training_domain"
    BOUNDARY = "boundary"
    OUTSIDE_TRAINING_DOMAIN = "outside_training_domain"
    INVALID = "invalid"


@dataclass(frozen=True)
class FeatureBounds:
    """The observed training range of one input feature.

    Attributes:
        name: The feature's name.
        lower: The minimum value observed in the training data.
        upper: The maximum value observed in the training data.
    """

    name: str
    lower: float
    upper: float

    @property
    def span(self) -> float:
        """The training range's width (``0.0`` for a constant feature)."""
        return self.upper - self.lower


@dataclass(frozen=True)
class DomainCheckResult:
    """The outcome of checking one query point against an :class:`ApplicabilityDomain`.

    Attributes:
        status: The overall :class:`DomainStatus`.
        per_feature_status: Each feature's individual status, keyed by name.
        out_of_bounds_features: The names of features that are outside
            their training range (empty unless ``status`` is
            :attr:`DomainStatus.OUTSIDE_TRAINING_DOMAIN`).
        warnings: Human-readable warnings describing the violation(s), if any.
    """

    status: DomainStatus
    per_feature_status: dict[str, DomainStatus] = field(default_factory=dict)
    out_of_bounds_features: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ApplicabilityDomain:
    """The training-data bounding box a surrogate's predictions should be trusted within.

    Attributes:
        feature_bounds: Each feature's observed training range, in the
            model's declared feature order.
    """

    feature_bounds: list[FeatureBounds]

    @classmethod
    def from_training_data(cls, x: np.ndarray, feature_names: list[str]) -> ApplicabilityDomain:
        """Build a domain from a training input matrix.

        Args:
            x: The training inputs, shape ``(n_samples, n_features)``.
            feature_names: The name of each column of ``x``, in order.

        Returns:
            A fitted :class:`ApplicabilityDomain`.
        """
        x = np.asarray(x, dtype=float)
        if x.shape[1] != len(feature_names):
            raise ValidationError(
                f"x has {x.shape[1]} columns but {len(feature_names)} feature_names were given."
            )
        bounds = [
            FeatureBounds(name=name, lower=float(x[:, i].min()), upper=float(x[:, i].max()))
            for i, name in enumerate(feature_names)
        ]
        return cls(feature_bounds=bounds)

    def check(self, point: dict[str, float]) -> DomainCheckResult:
        """Check one query point against this domain.

        Args:
            point: The query point's feature values, keyed by name.

        Returns:
            A :class:`DomainCheckResult`.
        """
        expected_names = [bounds.name for bounds in self.feature_bounds]
        missing = [name for name in expected_names if name not in point]
        extra = [name for name in point if name not in expected_names]
        if missing or extra:
            warnings = []
            if missing:
                warnings.append(f"Query point is missing feature(s): {missing}.")
            if extra:
                warnings.append(f"Query point carries unexpected feature(s): {extra}.")
            return DomainCheckResult(status=DomainStatus.INVALID, warnings=warnings)

        for name, value in point.items():
            if not np.isfinite(value):
                return DomainCheckResult(
                    status=DomainStatus.INVALID,
                    warnings=[f"Feature {name!r} has a non-finite value ({value!r})."],
                )

        per_feature: dict[str, DomainStatus] = {}
        out_of_bounds: list[str] = []
        warnings: list[str] = []
        for bounds in self.feature_bounds:
            value = float(point[bounds.name])
            if value < bounds.lower or value > bounds.upper:
                per_feature[bounds.name] = DomainStatus.OUTSIDE_TRAINING_DOMAIN
                out_of_bounds.append(bounds.name)
                warnings.append(
                    f"Feature {bounds.name!r}={value!r} lies outside its training range "
                    f"[{bounds.lower!r}, {bounds.upper!r}] (extrapolation)."
                )
                continue
            margin = BOUNDARY_FRACTION * bounds.span if bounds.span > 0.0 else 0.0
            if value <= bounds.lower + margin or value >= bounds.upper - margin:
                per_feature[bounds.name] = DomainStatus.BOUNDARY
            else:
                per_feature[bounds.name] = DomainStatus.WITHIN_TRAINING_DOMAIN

        if out_of_bounds:
            overall = DomainStatus.OUTSIDE_TRAINING_DOMAIN
        elif any(status is DomainStatus.BOUNDARY for status in per_feature.values()):
            overall = DomainStatus.BOUNDARY
        else:
            overall = DomainStatus.WITHIN_TRAINING_DOMAIN

        return DomainCheckResult(
            status=overall,
            per_feature_status=per_feature,
            out_of_bounds_features=out_of_bounds,
            warnings=warnings,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable representation of this domain."""
        return {
            "feature_bounds": [
                {"name": b.name, "lower": b.lower, "upper": b.upper} for b in self.feature_bounds
            ]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApplicabilityDomain:
        """Reconstruct an :class:`ApplicabilityDomain` from :meth:`to_dict`'s output."""
        return cls(
            feature_bounds=[
                FeatureBounds(name=item["name"], lower=item["lower"], upper=item["upper"])
                for item in data["feature_bounds"]
            ]
        )


__all__ = [
    "BOUNDARY_FRACTION",
    "ApplicabilityDomain",
    "DomainCheckResult",
    "DomainStatus",
    "FeatureBounds",
]
