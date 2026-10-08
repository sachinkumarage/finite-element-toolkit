"""Tests for femtoolkit.surrogate.rom.pod (Proper Orthogonal Decomposition)."""

from __future__ import annotations

import numpy as np
import pytest

from femtoolkit.exceptions import InvalidReducedBasisConfigurationError, ValidationError
from femtoolkit.surrogate.rom.pod import PODModel


def test_rank_one_matrix_captured_by_one_mode() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=25)
    b = rng.normal(size=6)
    snapshot_matrix = np.outer(a, b)

    model = PODModel().fit(snapshot_matrix, rank=1)
    assert model.rank == 1
    assert model.captured_energy > 1.0 - 1e-10

    reconstructed = model.predict(snapshot_matrix[:, 0])
    np.testing.assert_allclose(reconstructed, snapshot_matrix[:, 0], atol=1e-8)


def test_fixed_rank_selection() -> None:
    rng = np.random.default_rng(1)
    snapshot_matrix = rng.normal(size=(20, 10))
    model = PODModel().fit(snapshot_matrix, rank=4)
    assert model.rank == 4
    assert model.total_modes == 10
    assert 0.0 < model.captured_energy <= 1.0


def test_energy_threshold_selection_reaches_threshold() -> None:
    rng = np.random.default_rng(2)
    snapshot_matrix = rng.normal(size=(30, 15))
    model = PODModel().fit(snapshot_matrix, energy_threshold=0.95)
    assert model.captured_energy >= 0.95
    # One fewer mode must not reach the threshold (truncation is minimal).
    if model.rank > 1:
        fewer_modes_energy = float(np.sum(model.singular_values[: model.rank - 1] ** 2))
        total_energy = float(np.sum(model.singular_values**2))
        assert fewer_modes_energy / total_energy < 0.95


def test_no_truncation_retains_all_modes_and_full_energy() -> None:
    rng = np.random.default_rng(3)
    snapshot_matrix = rng.normal(size=(10, 6))
    model = PODModel().fit(snapshot_matrix)
    assert model.rank == model.total_modes
    assert model.captured_energy > 1.0 - 1e-10


def test_reduce_and_reconstruct_round_trip_with_full_rank() -> None:
    rng = np.random.default_rng(4)
    snapshot_matrix = rng.normal(size=(12, 8))
    model = PODModel().fit(snapshot_matrix)
    for i in range(snapshot_matrix.shape[1]):
        u = snapshot_matrix[:, i]
        q = model.reduce(u)
        reconstructed = model.reconstruct(q)
        np.testing.assert_allclose(reconstructed, u, atol=1e-8)


def test_reconstruction_error_reports_both_metrics() -> None:
    rng = np.random.default_rng(5)
    snapshot_matrix = rng.normal(size=(20, 10))
    model = PODModel().fit(snapshot_matrix, rank=2)
    error = model.error(snapshot_matrix[:, 0])
    assert error.absolute_error >= 0.0
    assert error.relative_error >= 0.0


def test_validate_against_held_out_snapshots() -> None:
    rng = np.random.default_rng(6)
    train_matrix = rng.normal(size=(20, 10))
    model = PODModel().fit(train_matrix, rank=5)

    held_out = rng.normal(size=(20, 3))
    report = model.validate(held_out)
    assert report.n_snapshots == 3
    assert report.rank == 5
    assert len(report.per_snapshot_errors) == 3
    assert report.max_relative_error >= report.mean_relative_error


def test_invalid_rank_rejected() -> None:
    rng = np.random.default_rng(7)
    snapshot_matrix = rng.normal(size=(10, 5))
    with pytest.raises(InvalidReducedBasisConfigurationError):
        PODModel().fit(snapshot_matrix, rank=0)
    with pytest.raises(InvalidReducedBasisConfigurationError):
        PODModel().fit(snapshot_matrix, rank=100)


def test_invalid_energy_threshold_rejected() -> None:
    rng = np.random.default_rng(8)
    snapshot_matrix = rng.normal(size=(10, 5))
    with pytest.raises(InvalidReducedBasisConfigurationError):
        PODModel().fit(snapshot_matrix, energy_threshold=1.5)


def test_rank_and_energy_threshold_mutually_exclusive() -> None:
    rng = np.random.default_rng(9)
    snapshot_matrix = rng.normal(size=(10, 5))
    with pytest.raises(ValidationError):
        PODModel().fit(snapshot_matrix, rank=1, energy_threshold=0.9)


def test_use_before_fit_raises() -> None:
    model = PODModel()
    with pytest.raises(ValidationError):
        model.reduce(np.zeros(5))
