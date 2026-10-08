"""Tests for femtoolkit.surrogate.persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from femtoolkit.exceptions import IncompatibleModelVersionError, ModelPersistenceError
from femtoolkit.surrogate.models.polynomial import PolynomialRegressionSurrogate
from femtoolkit.surrogate.models.rbf import RBFSurrogate
from femtoolkit.surrogate.persistence import (
    SURROGATE_PERSISTENCE_SCHEMA_VERSION,
    load_rom_model,
    load_surrogate_model,
    save_rom_model,
    save_surrogate_model,
)
from femtoolkit.surrogate.rom.pod import PODModel


def _fitted_polynomial() -> PolynomialRegressionSurrogate:
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, size=(20, 2))
    y = (x[:, 0] + 2.0 * x[:, 1]).reshape(-1, 1)
    model = PolynomialRegressionSurrogate(degree=1)
    model.fit(x, y, feature_names=["x1", "x2"], response_names=["y"], dataset_id="ds-1")
    return model


def test_save_and_load_polynomial_model_round_trip(tmp_path: Path) -> None:
    model = _fitted_polynomial()
    path = tmp_path / "model.json"
    save_surrogate_model(model, path)
    loaded = load_surrogate_model(path)

    query = {"x1": 0.3, "x2": -0.7}
    np.testing.assert_allclose(
        list(loaded.predict_point(query).values.values()),
        list(model.predict_point(query).values.values()),
    )
    assert loaded.training_metadata.dataset_id == "ds-1"


def test_save_and_load_rbf_model_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, size=(15, 1))
    y = np.sin(x)
    model = RBFSurrogate().fit(x, y, feature_names=["x"], response_names=["y"])

    path = tmp_path / "rbf.json"
    save_surrogate_model(model, path)
    loaded = load_surrogate_model(path)
    np.testing.assert_allclose(loaded.predict(x), model.predict(x))


def test_load_surrogate_model_rejects_corrupted_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not valid json {{{")
    with pytest.raises(ModelPersistenceError):
        load_surrogate_model(path)


def test_load_surrogate_model_rejects_incompatible_schema_version(tmp_path: Path) -> None:
    model = _fitted_polynomial()
    path = tmp_path / "model.json"
    save_surrogate_model(model, path)

    payload = json.loads(path.read_text())
    payload["schema_version"] = SURROGATE_PERSISTENCE_SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload))

    with pytest.raises(IncompatibleModelVersionError):
        load_surrogate_model(path)


def test_load_surrogate_model_rejects_missing_field(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"
    path.write_text(json.dumps({"schema_version": SURROGATE_PERSISTENCE_SCHEMA_VERSION}))
    with pytest.raises(ModelPersistenceError):
        load_surrogate_model(path)


def test_save_and_load_rom_model_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    snapshot_matrix = rng.normal(size=(20, 10))
    model = PODModel().fit(snapshot_matrix, rank=4)

    path = tmp_path / "pod.json"
    save_rom_model(model, path)
    loaded = load_rom_model(path)

    assert loaded.rank == model.rank
    np.testing.assert_allclose(loaded.captured_energy, model.captured_energy)
    np.testing.assert_allclose(loaded.basis, model.basis)


def test_load_rom_model_missing_array_file_raises(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    snapshot_matrix = rng.normal(size=(10, 5))
    model = PODModel().fit(snapshot_matrix, rank=2)
    path = tmp_path / "pod.json"
    save_rom_model(model, path)
    (path.with_suffix(path.suffix + ".npz")).unlink()

    with pytest.raises(ModelPersistenceError):
        load_rom_model(path)
