"""Tests for femtoolkit.surrogate.domain."""

from __future__ import annotations

import math

import numpy as np

from femtoolkit.surrogate.domain import ApplicabilityDomain, DomainStatus


def _domain() -> ApplicabilityDomain:
    x = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0], [3.0, 40.0], [4.0, 50.0]])
    return ApplicabilityDomain.from_training_data(x, ["a", "b"])


def test_point_within_domain() -> None:
    result = _domain().check({"a": 2.0, "b": 30.0})
    assert result.status is DomainStatus.WITHIN_TRAINING_DOMAIN
    assert result.out_of_bounds_features == []


def test_point_near_boundary() -> None:
    result = _domain().check({"a": 0.05, "b": 30.0})
    assert result.status is DomainStatus.BOUNDARY


def test_point_outside_domain() -> None:
    result = _domain().check({"a": 100.0, "b": 30.0})
    assert result.status is DomainStatus.OUTSIDE_TRAINING_DOMAIN
    assert "a" in result.out_of_bounds_features
    assert result.warnings


def test_invalid_point_missing_feature() -> None:
    result = _domain().check({"a": 1.0})
    assert result.status is DomainStatus.INVALID


def test_invalid_point_extra_feature() -> None:
    result = _domain().check({"a": 1.0, "b": 10.0, "c": 5.0})
    assert result.status is DomainStatus.INVALID


def test_invalid_point_non_finite_value() -> None:
    result = _domain().check({"a": math.nan, "b": 10.0})
    assert result.status is DomainStatus.INVALID


def test_domain_round_trip_serialization() -> None:
    domain = _domain()
    restored = ApplicabilityDomain.from_dict(domain.to_dict())
    assert [b.name for b in restored.feature_bounds] == [b.name for b in domain.feature_bounds]
    assert restored.check({"a": 2.0, "b": 30.0}).status is DomainStatus.WITHIN_TRAINING_DOMAIN
