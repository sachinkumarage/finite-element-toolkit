"""Tests for femtoolkit.surrogate.scaling."""

from __future__ import annotations

import numpy as np
import pytest

from femtoolkit.exceptions import InvalidScalingConfigurationError
from femtoolkit.surrogate.scaling import (
    IdentityScaler,
    MinMaxScaler,
    Scaler,
    StandardScaler,
    build_scaler,
)


def test_standard_scaler_round_trip() -> None:
    data = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-10)
    restored = scaler.inverse_transform(scaled)
    np.testing.assert_allclose(restored, data)


def test_minmax_scaler_round_trip() -> None:
    data = np.array([[1.0, -5.0], [5.0, 5.0]])
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    np.testing.assert_allclose(scaled.min(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(scaled.max(axis=0), 1.0, atol=1e-12)
    restored = scaler.inverse_transform(scaled)
    np.testing.assert_allclose(restored, data)


def test_zero_variance_feature_does_not_produce_nan() -> None:
    data = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    for scaler in (StandardScaler(), MinMaxScaler()):
        scaled = scaler.fit_transform(data)
        assert np.isfinite(scaled).all()


def test_transform_before_fit_raises() -> None:
    scaler = StandardScaler()
    with pytest.raises(InvalidScalingConfigurationError):
        scaler.transform(np.array([[1.0]]))


def test_identity_scaler_is_a_no_op() -> None:
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    scaler = IdentityScaler()
    np.testing.assert_allclose(scaler.fit_transform(data), data)
    np.testing.assert_allclose(scaler.inverse_transform(data), data)


def test_build_scaler_rejects_unknown_type() -> None:
    with pytest.raises(InvalidScalingConfigurationError):
        build_scaler("unknown")


def test_scaler_to_dict_from_dict_round_trip() -> None:
    data = np.array([[1.0, 100.0], [2.0, 50.0], [3.0, -10.0]])
    for original in (StandardScaler().fit(data), MinMaxScaler().fit(data)):
        restored = Scaler.from_dict(original.to_dict())
        np.testing.assert_allclose(restored.transform(data), original.transform(data))


def test_reproducible_fit_is_deterministic() -> None:
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    first = StandardScaler().fit_transform(data)
    second = StandardScaler().fit_transform(data)
    np.testing.assert_array_equal(first, second)
