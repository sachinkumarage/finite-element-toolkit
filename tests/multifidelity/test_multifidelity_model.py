"""Tests for femtoolkit.multifidelity.model (synthetic, no FEA)."""

from __future__ import annotations

import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.multifidelity.dataset import MultiFidelityDataset, MultiFidelitySample
from femtoolkit.multifidelity.discrepancy import train_discrepancy_surrogate
from femtoolkit.multifidelity.fidelity import HIGH_FIDELITY, LOW_FIDELITY, AnalyticalFidelityModel
from femtoolkit.multifidelity.model import (
    FusionAcceptanceStatus,
    MultiFidelityModel,
    verify_fused_prediction,
)
from femtoolkit.surrogate.models.polynomial import PolynomialRegressionSurrogate
from femtoolkit.surrogate.workflows.training import TrainingConfig


def _low_model() -> AnalyticalFidelityModel:
    return AnalyticalFidelityModel(
        name="identity", evaluate_fn=lambda point: {"y": point["x"]}, level=LOW_FIDELITY
    )


def _high_model() -> AnalyticalFidelityModel:
    return AnalyticalFidelityModel(
        name="true", evaluate_fn=lambda point: {"y": point["x"] + 2.0 * point["x"]},
        level=HIGH_FIDELITY,
    )


def _mf_model(discrepancy_model) -> MultiFidelityModel:
    return MultiFidelityModel(low_fidelity_model=_low_model(), discrepancy_model=discrepancy_model)


def _fitted_dataset_and_discrepancy_model():
    dataset = MultiFidelityDataset(feature_names=["x"], response_names=["y"])
    low, high = _low_model(), _high_model()
    for i in range(10):
        point = {"x": float(i)}
        dataset.add_sample(
            MultiFidelitySample(
                sample_id=str(i), inputs=point, low_result=low.evaluate(point),
                high_result=high.evaluate(point),
            )
        )
    model, _report = train_discrepancy_surrogate(
        dataset, TrainingConfig(model_type="polynomial", model_kwargs={"degree": 1}, split_seed=0)
    )
    return dataset, model


def test_multi_fidelity_model_requires_fitted_discrepancy_model() -> None:
    unfitted = PolynomialRegressionSurrogate(degree=1)
    with pytest.raises(ValidationError):
        MultiFidelityModel(low_fidelity_model=_low_model(), discrepancy_model=unfitted)


def test_multi_fidelity_model_keeps_components_separate() -> None:
    _dataset, discrepancy_model = _fitted_dataset_and_discrepancy_model()
    mf_model = _mf_model(discrepancy_model)

    prediction = mf_model.predict({"x": 5.0})
    assert prediction.low_fidelity_result["y"] == pytest.approx(5.0)
    assert prediction.predicted_discrepancy["y"] == pytest.approx(10.0, abs=1e-6)
    assert prediction.fused_prediction["y"] == pytest.approx(15.0, abs=1e-6)
    assert prediction.is_fused_prediction is True


def test_verify_fused_prediction_reports_improved_status() -> None:
    _dataset, discrepancy_model = _fitted_dataset_and_discrepancy_model()
    mf_model = _mf_model(discrepancy_model)
    records = verify_fused_prediction(mf_model, _high_model(), [{"x": 12.0}])

    record = records[0]
    assert record.status is FusionAcceptanceStatus.IMPROVED
    assert record.high_fidelity_result["y"] == pytest.approx(36.0)
    assert record.fused_absolute_error["y"] < record.low_fidelity_absolute_error["y"]


def test_verify_fused_prediction_handles_failed_high_fidelity_evaluation() -> None:
    _dataset, discrepancy_model = _fitted_dataset_and_discrepancy_model()
    mf_model = _mf_model(discrepancy_model)
    failing_high_model = AnalyticalFidelityModel(
        name="failing", evaluate_fn=lambda point: {}, level=HIGH_FIDELITY
    )
    records = verify_fused_prediction(mf_model, failing_high_model, [{"x": 1.0}])
    assert records[0].status is FusionAcceptanceStatus.VERIFICATION_FAILED
    assert records[0].high_fidelity_result == {}
