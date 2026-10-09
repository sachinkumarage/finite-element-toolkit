"""Measured/reference engineering data for digital twin model updating (Version 38).

**Engineering concept.** A digital twin compares a simulation's prediction
against observed physical behavior -- a measured displacement, temperature,
strain, natural frequency, stress, or load response. This module stores that
measured data as a lightweight, in-memory dataset; it is not a sensor-streaming
system or a measurement database (see the module docstring of
:mod:`femtoolkit.digital_twin` for the full scope boundary).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from femtoolkit.exceptions import ValidationError


@dataclass
class MeasurementPoint:
    """One measured observation of an engineering quantity.

    Attributes:
        measurement_id: A unique identifier for this measurement.
        input_conditions: The conditions this measurement was taken under
            (e.g. applied load, temperature), keyed by the same dotted
            override path :class:`~femtoolkit.studies.scenarios.Scenario`
            uses. Empty if the measurement was taken under the simulation
            model's own default conditions.
        measured_value: The observed value, in physical units.
        uncertainty: An optional measurement uncertainty (standard deviation
            or similar), in the same units as ``measured_value``. ``None`` if
            not characterized.
        metadata: Free-form additional information (e.g. who took the
            measurement, which sensor, synthetic-data generation settings).
        created_at: ISO-8601 UTC timestamp when this point was recorded.
    """

    measurement_id: str
    input_conditions: dict[str, float] = field(default_factory=dict)
    measured_value: float = 0.0
    uncertainty: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        if not np.isfinite(self.measured_value):
            raise ValidationError(
                f"Measurement {self.measurement_id!r} has a non-finite measured_value "
                f"({self.measured_value!r})."
            )
        if self.uncertainty is not None and self.uncertainty < 0.0:
            raise ValidationError(
                f"Measurement {self.measurement_id!r} uncertainty must be non-negative, "
                f"got {self.uncertainty!r}."
            )
        for name, value in self.input_conditions.items():
            if not np.isfinite(value):
                raise ValidationError(
                    f"Measurement {self.measurement_id!r} input condition {name!r} has a "
                    f"non-finite value ({value!r})."
                )


@dataclass
class MeasurementData:
    """A named engineering quantity's measured reference data.

    Attributes:
        quantity_name: The measured quantity's name -- must be one of
            :data:`~femtoolkit.studies.extractors.EXTRACTORS`, so the same
            named extractor already used throughout this toolkit reads the
            matching simulation prediction.
        units: A units string, purely descriptive.
        points: Every measured observation, in insertion order.
        dataset_id: A unique, stable identifier for this dataset.
        description: A short, human-readable description (e.g. "synthetic
            data generated for the cantilever beam example").
    """

    quantity_name: str
    units: str = ""
    points: list[MeasurementPoint] = field(default_factory=list)
    dataset_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""

    def __post_init__(self) -> None:
        if not self.quantity_name:
            raise ValidationError("MeasurementData requires a non-empty quantity_name.")
        validated = list(self.points)
        self.points = []
        for point in validated:
            self.add_point(point)

    def add_point(self, point: MeasurementPoint) -> None:
        """Validate and append one measurement point to this dataset, in place.

        Args:
            point: The measurement point to add.

        Raises:
            ValidationError: If ``point.measurement_id`` duplicates an existing one.
        """
        existing_ids = {existing.measurement_id for existing in self.points}
        if point.measurement_id in existing_ids:
            raise ValidationError(
                f"Measurement ID {point.measurement_id!r} already exists in this dataset."
            )
        self.points.append(point)

    @property
    def n_points(self) -> int:
        """How many measurement points this dataset holds."""
        return len(self.points)

    def measured_values(self) -> np.ndarray:
        """Every point's measured value, as an array, in :attr:`points` order."""
        return np.array([point.measured_value for point in self.points], dtype=float)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable representation of this dataset."""
        return {
            "quantity_name": self.quantity_name,
            "units": self.units,
            "dataset_id": self.dataset_id,
            "description": self.description,
            "points": [
                {
                    "measurement_id": point.measurement_id,
                    "input_conditions": dict(point.input_conditions),
                    "measured_value": point.measured_value,
                    "uncertainty": point.uncertainty,
                    "metadata": dict(point.metadata),
                    "created_at": point.created_at,
                }
                for point in self.points
            ],
        }


__all__ = ["MeasurementData", "MeasurementPoint"]
