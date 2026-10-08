"""Tests for femtoolkit.surrogate.models (polynomial regression and RBF surrogates)."""

from __future__ import annotations

import numpy as np
import pytest

from femtoolkit.exceptions import SurrogateFittingError, ValidationError
from femtoolkit.surrogate.domain import DomainStatus
from femtoolkit.surrogate.models.base import PredictionStatus
from femtoolkit.surrogate.models.polynomial import PolynomialRegressionSurrogate
from femtoolkit.surrogate.models.rbf import RBFSurrogate


def _quadratic_dataset(n: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=(n, 2))
    y = (2.0 + 3.0 * x[:, 0] - x[:, 1] + 0.5 * x[:, 0] ** 2 + x[:, 0] * x[:, 1]).reshape(-1, 1)
    return x, y


def test_polynomial_reproduces_known_quadratic_function() -> None:
    x, y = _quadratic_dataset()
    model = PolynomialRegressionSurrogate(degree=2)
    model.fit(x, y, feature_names=["x1", "x2"], response_names=["y"])
    predicted = model.predict(x)
    np.testing.assert_allclose(predicted, y, atol=1e-6)


def test_polynomial_degree_one_ignores_interaction_terms() -> None:
    x, y = _quadratic_dataset()
    model = PolynomialRegressionSurrogate(degree=1)
    model.fit(x, y, feature_names=["x1", "x2"], response_names=["y"])
    predicted = model.predict(x)
    # A linear model cannot reproduce the quadratic/cross terms exactly.
    assert not np.allclose(predicted, y, atol=1e-6)


def test_polynomial_rejects_unsupported_degree() -> None:
    with pytest.raises(SurrogateFittingError):
        PolynomialRegressionSurrogate(degree=3)


def test_polynomial_requires_enough_points_for_coefficients() -> None:
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0]])
    model = PolynomialRegressionSurrogate(degree=2)
    with pytest.raises(SurrogateFittingError):
        model.fit(x, y, feature_names=["x1", "x2"], response_names=["y"])


def test_predict_point_returns_physical_units_and_domain_status() -> None:
    x, y = _quadratic_dataset()
    model = PolynomialRegressionSurrogate(degree=2)
    model.fit(
        x, y, feature_names=["x1", "x2"], response_names=["y"], dataset_id="ds-1", dataset_version=1
    )
    prediction = model.predict_point({"x1": 0.0, "x2": 0.0})
    assert prediction.status is PredictionStatus.OK
    assert prediction.domain_status is DomainStatus.WITHIN_TRAINING_DOMAIN
    assert prediction.is_surrogate_prediction is True
    assert prediction.dataset_id == "ds-1"
    np.testing.assert_allclose(prediction.values["y"], 2.0, atol=1e-6)


def test_predict_point_detects_missing_feature() -> None:
    x, y = _quadratic_dataset()
    model = PolynomialRegressionSurrogate(degree=1)
    model.fit(x, y, feature_names=["x1", "x2"], response_names=["y"])
    prediction = model.predict_point({"x1": 0.0})
    assert prediction.status is PredictionStatus.INVALID_INPUT


def test_predict_point_detects_out_of_domain() -> None:
    x, y = _quadratic_dataset()
    model = PolynomialRegressionSurrogate(degree=1)
    model.fit(x, y, feature_names=["x1", "x2"], response_names=["y"])
    prediction = model.predict_point({"x1": 1000.0, "x2": 1000.0})
    assert prediction.domain_status is DomainStatus.OUTSIDE_TRAINING_DOMAIN


def test_predict_before_fit_raises() -> None:
    model = PolynomialRegressionSurrogate()
    with pytest.raises(ValidationError):
        model.predict(np.array([[0.0, 0.0]]))


def test_rbf_reproduces_known_function() -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(-2.0, 2.0, size=(30, 1))
    y = np.sin(x)
    model = RBFSurrogate(kernel="gaussian")
    model.fit(x, y, feature_names=["x"], response_names=["y"])
    predicted = model.predict(x)
    np.testing.assert_allclose(predicted, y, atol=1e-3)


def test_rbf_multiquadric_kernel_is_numerically_stable() -> None:
    rng = np.random.default_rng(2)
    x = rng.uniform(-1.0, 1.0, size=(25, 2))
    y = (x[:, 0] ** 2 + x[:, 1] ** 2).reshape(-1, 1)
    model = RBFSurrogate(kernel="multiquadric", regularization=1e-6)
    model.fit(x, y, feature_names=["x1", "x2"], response_names=["y"])
    predicted = model.predict(x)
    assert np.isfinite(predicted).all()
    np.testing.assert_allclose(predicted, y, atol=1e-2)


def test_rbf_rejects_unknown_kernel() -> None:
    with pytest.raises(SurrogateFittingError):
        RBFSurrogate(kernel="unknown")


def test_rbf_reproducible_across_identical_fits() -> None:
    rng = np.random.default_rng(3)
    x = rng.uniform(-1.0, 1.0, size=(20, 1))
    y = np.sin(x)
    first = RBFSurrogate().fit(x, y, feature_names=["x"], response_names=["y"]).predict(x)
    second = RBFSurrogate().fit(x, y, feature_names=["x"], response_names=["y"]).predict(x)
    np.testing.assert_array_equal(first, second)


def test_validate_returns_metric_set_per_response() -> None:
    x, y = _quadratic_dataset()
    model = PolynomialRegressionSurrogate(degree=2)
    model.fit(x, y, feature_names=["x1", "x2"], response_names=["y"])
    metrics = model.validate(x, y)
    assert "y" in metrics
    assert metrics["y"].r2 > 0.99
