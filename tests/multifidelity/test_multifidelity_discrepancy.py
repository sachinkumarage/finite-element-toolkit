"""Tests for femtoolkit.multifidelity.discrepancy."""

from __future__ import annotations

import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.multifidelity.dataset import MultiFidelityDataset, MultiFidelitySample
from femtoolkit.multifidelity.discrepancy import train_discrepancy_surrogate
from femtoolkit.surrogate.workflows.training import TrainingConfig


def _linear_discrepancy_dataset(n: int = 12) -> MultiFidelityDataset:
    dataset = MultiFidelityDataset(feature_names=["x"], response_names=["y"])
    for i in range(n):
        x = float(i)
        dataset.add_sample(
            MultiFidelitySample(
                sample_id=str(i), inputs={"x": x}, low_result={"y": x},
                high_result={"y": x + 2.0 * x},
            )
        )
    return dataset


def test_train_discrepancy_surrogate_fits_a_linear_discrepancy() -> None:
    dataset = _linear_discrepancy_dataset()
    model, report = train_discrepancy_surrogate(
        dataset, TrainingConfig(model_type="polynomial", model_kwargs={"degree": 1}, split_seed=0)
    )
    assert model.is_fitted
    assert report.test_metrics["y"].r2 > 0.99

    prediction = model.predict_point({"x": 5.0})
    assert prediction.values["y"] == pytest.approx(10.0, abs=1e-6)


def test_train_discrepancy_surrogate_requires_paired_samples() -> None:
    dataset = MultiFidelityDataset(feature_names=["x"], response_names=["y"])
    dataset.add_sample(MultiFidelitySample(sample_id="a", inputs={"x": 1.0}, low_result={"y": 1.0}))
    with pytest.raises(ValidationError):
        train_discrepancy_surrogate(dataset)


def test_train_discrepancy_surrogate_is_reproducible_with_same_seed() -> None:
    dataset = _linear_discrepancy_dataset()
    config = TrainingConfig(model_type="polynomial", model_kwargs={"degree": 1}, split_seed=0)
    model_a, _ = train_discrepancy_surrogate(dataset, config)
    model_b, _ = train_discrepancy_surrogate(dataset, config)
    point = {"x": 7.0}
    assert model_a.predict_point(point).values == model_b.predict_point(point).values
