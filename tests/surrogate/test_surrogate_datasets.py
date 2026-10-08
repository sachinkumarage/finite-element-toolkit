"""Tests for femtoolkit.surrogate.datasets."""

from __future__ import annotations

import math

import pytest

from femtoolkit.exceptions import (
    InconsistentSnapshotError,
    InsufficientSnapshotsError,
    ValidationError,
)
from femtoolkit.surrogate.datasets import Snapshot, SnapshotDataset


def _snapshot(snapshot_id: str, x1: float, x2: float, y: float) -> Snapshot:
    return Snapshot(snapshot_id=snapshot_id, inputs={"x1": x1, "x2": x2}, outputs={"y": y})


def _dataset(n: int = 10) -> SnapshotDataset:
    snapshots = [_snapshot(str(i), float(i), float(i) * 2, float(i) ** 2) for i in range(n)]
    return SnapshotDataset(feature_names=["x1", "x2"], response_names=["y"], snapshots=snapshots)


def test_add_snapshot_validates_dimensions() -> None:
    dataset = SnapshotDataset(feature_names=["x1"], response_names=["y"])
    dataset.add_snapshot(Snapshot(snapshot_id="a", inputs={"x1": 1.0}, outputs={"y": 2.0}))
    assert dataset.n_samples == 1

    with pytest.raises(InconsistentSnapshotError):
        dataset.add_snapshot(
            Snapshot(snapshot_id="b", inputs={"x1": 1.0, "x2": 9.0}, outputs={"y": 2.0})
        )

    with pytest.raises(InconsistentSnapshotError):
        dataset.add_snapshot(Snapshot(snapshot_id="c", inputs={"x1": 1.0}, outputs={}))

    with pytest.raises(InconsistentSnapshotError):
        dataset.add_snapshot(Snapshot(snapshot_id="d", inputs={"x1": math.nan}, outputs={"y": 2.0}))


def test_feature_response_names_must_be_disjoint() -> None:
    with pytest.raises(ValidationError):
        SnapshotDataset(feature_names=["x1"], response_names=["x1"])


def test_to_arrays_preserves_order() -> None:
    dataset = _dataset(5)
    x, y = dataset.to_arrays()
    assert x.shape == (5, 2)
    assert y.shape == (5, 1)
    for i in range(5):
        assert x[i, 0] == float(i)
        assert y[i, 0] == float(i) ** 2


def test_with_additional_snapshots_bumps_version_without_mutating_original() -> None:
    dataset = _dataset(5)
    extra = [_snapshot("extra-0", 100.0, 200.0, 300.0)]
    updated = dataset.with_additional_snapshots(extra)

    assert dataset.dataset_version == 1
    assert dataset.n_samples == 5
    assert updated.dataset_version == 2
    assert updated.n_samples == 6
    assert updated.dataset_id == dataset.dataset_id


def test_split_sizes_and_reproducibility() -> None:
    dataset = _dataset(20)
    split_a = dataset.split(train_fraction=0.7, validation_fraction=0.15, seed=42)
    split_b = dataset.split(train_fraction=0.7, validation_fraction=0.15, seed=42)

    assert split_a.train.n_samples + split_a.validation.n_samples + split_a.test.n_samples == 20
    assert split_a.test.n_samples >= 1
    assert [s.snapshot_id for s in split_a.train.snapshots] == [
        s.snapshot_id for s in split_b.train.snapshots
    ]


def test_split_small_dataset_has_no_validation_subset() -> None:
    dataset = _dataset(5)
    split = dataset.split(seed=0)
    assert split.validation.n_samples == 0
    assert split.train.n_samples + split.test.n_samples == 5
    assert split.test.n_samples >= 1


def test_split_requires_minimum_snapshots() -> None:
    dataset = _dataset(2)
    with pytest.raises(InsufficientSnapshotsError):
        dataset.split(seed=0)


def test_to_dict_from_dict_round_trip() -> None:
    dataset = _dataset(4)
    restored = SnapshotDataset.from_dict(dataset.to_dict())
    assert restored.feature_names == dataset.feature_names
    assert restored.response_names == dataset.response_names
    assert restored.n_samples == dataset.n_samples
    assert restored.dataset_id == dataset.dataset_id
