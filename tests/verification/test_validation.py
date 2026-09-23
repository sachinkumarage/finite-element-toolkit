"""Tests for femtoolkit.validation (Version 29)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.validation import (
    ReferenceDataset,
    compare_to_reference_dataset,
    load_reference_dataset_csv,
    load_reference_dataset_json,
    reference_dataset_from_dict,
)
from femtoolkit.verification.status import VerificationStatus
from femtoolkit.verification.tolerance import Tolerance


def _dataset(values=(0.0, 0.5, 1.0)) -> ReferenceDataset:
    return ReferenceDataset(
        name="Test dataset",
        source="self-generated: unit test fixture",
        quantity="Displacement",
        units="m",
        independent_variable=np.array([0.0, 1.0, 2.0]),
        independent_variable_label="x (m)",
        values=np.array(values),
    )


def test_reference_dataset_rejects_blank_name() -> None:
    with pytest.raises(ValidationError):
        ReferenceDataset(
            name="   ",
            source="s",
            quantity="q",
            units="m",
            independent_variable=np.array([0.0]),
            independent_variable_label="x",
            values=np.array([0.0]),
        )


def test_reference_dataset_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValidationError):
        ReferenceDataset(
            name="n",
            source="s",
            quantity="q",
            units="m",
            independent_variable=np.array([0.0, 1.0]),
            independent_variable_label="x",
            values=np.array([0.0]),
        )


def test_reference_dataset_rejects_mismatched_uncertainty_length() -> None:
    with pytest.raises(ValidationError):
        ReferenceDataset(
            name="n",
            source="s",
            quantity="q",
            units="m",
            independent_variable=np.array([0.0, 1.0]),
            independent_variable_label="x",
            values=np.array([0.0, 1.0]),
            uncertainty=np.array([0.1]),
        )


def test_compare_to_reference_dataset_passes_for_close_match() -> None:
    dataset = _dataset()
    result = compare_to_reference_dataset(dataset, np.array([0.001, 0.499, 1.002]))
    assert result.status is VerificationStatus.PASS


def test_compare_to_reference_dataset_fails_for_far_mismatch() -> None:
    dataset = _dataset()
    result = compare_to_reference_dataset(
        dataset, np.array([5.0, 5.0, 5.0]), tolerance=Tolerance(absolute=1e-9, relative=1e-9)
    )
    assert result.status is VerificationStatus.FAIL


def test_compare_to_reference_dataset_rejects_shape_mismatch() -> None:
    dataset = _dataset()
    with pytest.raises(ValidationError):
        compare_to_reference_dataset(dataset, np.array([0.0, 0.5]))


def test_compare_result_records_source_honestly() -> None:
    dataset = _dataset()
    result = compare_to_reference_dataset(dataset, np.array([0.0, 0.5, 1.0]))
    assert result.dataset_source == "self-generated: unit test fixture"
    assert "self-generated" in result.message


def test_compare_to_reference_dataset_reports_uncertainty_presence() -> None:
    with_uncertainty = ReferenceDataset(
        name="n",
        source="s",
        quantity="q",
        units="m",
        independent_variable=np.array([0.0]),
        independent_variable_label="x",
        values=np.array([1.0]),
        uncertainty=np.array([0.1]),
    )
    result = compare_to_reference_dataset(with_uncertainty, np.array([1.0]))
    assert result.has_uncertainty is True

    without_uncertainty = _dataset()
    result_2 = compare_to_reference_dataset(without_uncertainty, np.array([0.0, 0.5, 1.0]))
    assert result_2.has_uncertainty is False


def test_reference_dataset_from_dict_round_trip() -> None:
    data = {
        "name": "Dict Test",
        "source": "self-generated",
        "quantity": "Temperature",
        "units": "K",
        "independent_variable": [0.0, 1.0],
        "values": [300.0, 310.0],
    }
    dataset = reference_dataset_from_dict(data)
    assert dataset.name == "Dict Test"
    np.testing.assert_array_equal(dataset.values, np.array([300.0, 310.0]))


def test_reference_dataset_from_dict_missing_key_raises() -> None:
    with pytest.raises(ValidationError):
        reference_dataset_from_dict({"name": "Incomplete"})


def test_load_reference_dataset_json_round_trip(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(
        json.dumps(
            {
                "name": "JSON Dataset",
                "source": "self-generated",
                "quantity": "Displacement",
                "units": "m",
                "independent_variable": [0.0, 1.0, 2.0],
                "values": [0.0, 0.5, 1.0],
            }
        )
    )
    dataset = load_reference_dataset_json(path)
    assert dataset.name == "JSON Dataset"
    assert len(dataset.values) == 3


def test_load_reference_dataset_json_missing_file_raises(tmp_path) -> None:
    with pytest.raises(ValidationError):
        load_reference_dataset_json(tmp_path / "does_not_exist.json")


def test_load_reference_dataset_json_invalid_json_raises(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not valid json")
    with pytest.raises(ValidationError):
        load_reference_dataset_json(path)


def test_load_reference_dataset_csv_round_trip(tmp_path) -> None:
    path = tmp_path / "dataset.csv"
    path.write_text("x,value\n0.0,0.0\n1.0,0.5\n2.0,1.0\n")
    dataset = load_reference_dataset_csv(
        path, name="CSV Dataset", source="self-generated", quantity="Displacement", units="m"
    )
    assert dataset.name == "CSV Dataset"
    np.testing.assert_array_equal(dataset.values, np.array([0.0, 0.5, 1.0]))
    assert dataset.uncertainty is None


def test_load_reference_dataset_csv_with_uncertainty_column(tmp_path) -> None:
    path = tmp_path / "dataset_with_uncertainty.csv"
    path.write_text("x,value,uncertainty\n0.0,0.0,0.01\n1.0,0.5,0.02\n")
    dataset = load_reference_dataset_csv(
        path, name="CSV Dataset", source="self-generated", quantity="Displacement", units="m"
    )
    assert dataset.uncertainty is not None
    np.testing.assert_array_equal(dataset.uncertainty, np.array([0.01, 0.02]))


def test_load_reference_dataset_csv_missing_file_raises(tmp_path) -> None:
    with pytest.raises(ValidationError):
        load_reference_dataset_csv(
            tmp_path / "missing.csv", name="n", source="s", quantity="q", units="m"
        )


def test_load_reference_dataset_csv_non_numeric_value_raises(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("x,value\n0.0,not_a_number\n")
    with pytest.raises(ValidationError):
        load_reference_dataset_csv(path, name="n", source="s", quantity="q", units="m")
