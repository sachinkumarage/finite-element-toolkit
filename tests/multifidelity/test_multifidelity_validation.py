"""Tests for femtoolkit.multifidelity.validation (synthetic, no FEA)."""

from __future__ import annotations

import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.multifidelity.dataset import MultiFidelityDataset, MultiFidelitySample
from femtoolkit.multifidelity.discrepancy import train_discrepancy_surrogate
from femtoolkit.multifidelity.fidelity import HIGH_FIDELITY, LOW_FIDELITY, AnalyticalFidelityModel
from femtoolkit.multifidelity.model import MultiFidelityModel
from femtoolkit.multifidelity.validation import compare_fidelity_accuracy
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


def _dataset() -> MultiFidelityDataset:
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
    return dataset


def test_compare_fidelity_accuracy_shows_fusion_improves_on_linear_discrepancy() -> None:
    dataset = _dataset()
    discrepancy_model, _report = train_discrepancy_surrogate(
        dataset, TrainingConfig(model_type="polynomial", model_kwargs={"degree": 1}, split_seed=0)
    )
    mf_model = _mf_model(discrepancy_model)

    comparison = compare_fidelity_accuracy(mf_model, dataset)
    assert comparison.n_samples == dataset.n_samples
    assert comparison.fused_metrics["y"].rmse < comparison.low_fidelity_metrics["y"].rmse
    assert comparison.improves_on_low_fidelity("y") is True


def test_compare_fidelity_accuracy_requires_paired_samples() -> None:
    dataset = MultiFidelityDataset(feature_names=["x"], response_names=["y"])
    dataset.add_sample(MultiFidelitySample(sample_id="a", inputs={"x": 1.0}, low_result={"y": 1.0}))
    discrepancy_dataset = _dataset()
    discrepancy_model, _report = train_discrepancy_surrogate(discrepancy_dataset)
    mf_model = _mf_model(discrepancy_model)
    with pytest.raises(ValidationError):
        compare_fidelity_accuracy(mf_model, dataset)


def test_improves_on_low_fidelity_rejects_unknown_response() -> None:
    dataset = _dataset()
    discrepancy_model, _report = train_discrepancy_surrogate(dataset)
    mf_model = _mf_model(discrepancy_model)
    comparison = compare_fidelity_accuracy(mf_model, dataset)
    with pytest.raises(ValidationError):
        comparison.improves_on_low_fidelity("unknown")
