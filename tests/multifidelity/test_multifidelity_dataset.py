"""Tests for femtoolkit.multifidelity.dataset."""

from __future__ import annotations

import pytest

from femtoolkit.exceptions import IncompatibleFidelityResultError, ValidationError
from femtoolkit.multifidelity.dataset import (
    MultiFidelityDataset,
    MultiFidelitySample,
    compute_discrepancy,
)


def _dataset() -> MultiFidelityDataset:
    return MultiFidelityDataset(feature_names=["x"], response_names=["y"])


def _paired_sample(sample_id: str, x: float, low: float, high: float) -> MultiFidelitySample:
    return MultiFidelitySample(
        sample_id=sample_id, inputs={"x": x}, low_result={"y": low}, high_result={"y": high}
    )


def test_compute_discrepancy_matches_formula() -> None:
    delta = compute_discrepancy({"y": 2.0}, {"y": 5.0})
    assert delta == {"y": 3.0}


def test_compute_discrepancy_rejects_mismatched_responses() -> None:
    with pytest.raises(IncompatibleFidelityResultError):
        compute_discrepancy({"y": 2.0}, {"z": 5.0})


def test_compute_discrepancy_rejects_non_finite_values() -> None:
    with pytest.raises(IncompatibleFidelityResultError):
        compute_discrepancy({"y": float("nan")}, {"y": 5.0})


def test_add_sample_accepts_paired_and_unpaired() -> None:
    dataset = _dataset()
    dataset.add_sample(_paired_sample("a", 1.0, 1.0, 1.5))
    dataset.add_sample(MultiFidelitySample(sample_id="b", inputs={"x": 2.0}, low_result={"y": 2.0}))
    assert dataset.n_samples == 2
    assert len(dataset.paired_samples()) == 1
    assert len(dataset.unpaired_samples()) == 1


def test_add_sample_rejects_wrong_input_dimensions() -> None:
    dataset = _dataset()
    with pytest.raises(ValidationError):
        dataset.add_sample(
            MultiFidelitySample(
                sample_id="a", inputs={"x": 1.0, "extra": 2.0}, low_result={"y": 1.0}
            )
        )


def test_add_sample_rejects_incompatible_high_result() -> None:
    dataset = _dataset()
    with pytest.raises(IncompatibleFidelityResultError):
        dataset.add_sample(
            MultiFidelitySample(
                sample_id="a", inputs={"x": 1.0}, low_result={"y": 1.0}, high_result={"z": 1.5}
            )
        )


def test_sample_discrepancy_requires_pairing() -> None:
    sample = MultiFidelitySample(sample_id="a", inputs={"x": 1.0}, low_result={"y": 1.0})
    with pytest.raises(ValidationError):
        sample.discrepancy()


def test_discrepancies_computed_in_order_over_paired_samples_only() -> None:
    dataset = _dataset()
    dataset.add_sample(_paired_sample("a", 1.0, 1.0, 2.0))
    dataset.add_sample(MultiFidelitySample(sample_id="b", inputs={"x": 2.0}, low_result={"y": 2.0}))
    dataset.add_sample(_paired_sample("c", 3.0, 3.0, 4.5))
    deltas = dataset.discrepancies()
    assert list(deltas["y"]) == [1.0, 1.5]


def test_to_discrepancy_dataset_builds_a_snapshot_dataset() -> None:
    dataset = _dataset()
    dataset.add_sample(_paired_sample("a", 1.0, 1.0, 2.0))
    dataset.add_sample(_paired_sample("b", 2.0, 2.0, 4.0))
    discrepancy_dataset = dataset.to_discrepancy_dataset()
    assert discrepancy_dataset.feature_names == ["x"]
    assert discrepancy_dataset.response_names == ["y"]
    assert discrepancy_dataset.n_samples == 2
    outputs = sorted(snapshot.outputs["y"] for snapshot in discrepancy_dataset.snapshots)
    assert outputs == [1.0, 2.0]


def test_to_discrepancy_dataset_requires_a_paired_sample() -> None:
    dataset = _dataset()
    dataset.add_sample(MultiFidelitySample(sample_id="a", inputs={"x": 1.0}, low_result={"y": 1.0}))
    with pytest.raises(ValidationError):
        dataset.to_discrepancy_dataset()
