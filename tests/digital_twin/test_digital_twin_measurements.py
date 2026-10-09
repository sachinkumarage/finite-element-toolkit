"""Tests for femtoolkit.digital_twin.measurements."""

from __future__ import annotations

import pytest

from femtoolkit.digital_twin.measurements import MeasurementData, MeasurementPoint
from femtoolkit.exceptions import ValidationError


def test_valid_measurement_data() -> None:
    data = MeasurementData(quantity_name="maximum_displacement", units="m")
    data.add_point(MeasurementPoint(measurement_id="m1", measured_value=0.003))
    data.add_point(MeasurementPoint(measurement_id="m2", measured_value=0.004))
    assert data.n_points == 2
    assert list(data.measured_values()) == [0.003, 0.004]


def test_measurement_data_requires_quantity_name() -> None:
    with pytest.raises(ValidationError):
        MeasurementData(quantity_name="")


def test_duplicate_measurement_id_rejected() -> None:
    data = MeasurementData(quantity_name="maximum_displacement")
    data.add_point(MeasurementPoint(measurement_id="m1", measured_value=0.003))
    with pytest.raises(ValidationError):
        data.add_point(MeasurementPoint(measurement_id="m1", measured_value=0.004))


def test_measurement_point_rejects_non_finite_measured_value() -> None:
    with pytest.raises(ValidationError):
        MeasurementPoint(measurement_id="m1", measured_value=float("nan"))


def test_measurement_point_rejects_negative_uncertainty() -> None:
    with pytest.raises(ValidationError):
        MeasurementPoint(measurement_id="m1", measured_value=1.0, uncertainty=-0.1)


def test_measurement_point_rejects_non_finite_input_condition() -> None:
    with pytest.raises(ValidationError):
        MeasurementPoint(
            measurement_id="m1", measured_value=1.0, input_conditions={"load": float("inf")}
        )


def test_measurement_point_metadata_and_uncertainty_roundtrip() -> None:
    point = MeasurementPoint(
        measurement_id="m1", measured_value=1.0, uncertainty=0.05,
        input_conditions={"loads.0.magnitude": -2000.0}, metadata={"sensor": "strain-gauge-1"},
    )
    data = MeasurementData(quantity_name="maximum_displacement")
    data.add_point(point)
    payload = data.to_dict()
    assert payload["points"][0]["uncertainty"] == 0.05
    assert payload["points"][0]["metadata"] == {"sensor": "strain-gauge-1"}
    assert payload["points"][0]["input_conditions"] == {"loads.0.magnitude": -2000.0}
