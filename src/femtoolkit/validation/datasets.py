"""Reference/experimental dataset infrastructure for FEA validation (Version 29).

:class:`ReferenceDataset` is deliberately **infrastructure only**: this
module provides a structured, metadata-carrying container for a
trusted reference dataset and loaders for the two common serialization
formats (CSV, JSON), but it ships with **no bundled experimental data**
and fabricates none. A dataset always comes from the caller -- read
from a file the caller supplies, or built in memory from values the
caller already trusts. If no such dataset exists for a given quantity,
the honest answer is that *validation* (as opposed to *verification*)
has not been performed for it; see the module docstring of
:mod:`femtoolkit.validation` and ``docs/validation.md`` for the
verification/validation distinction this toolkit enforces throughout
its API and reports.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from femtoolkit.exceptions import ValidationError


@dataclass(frozen=True)
class ReferenceDataset:
    """A trusted reference dataset to compare simulation results against.

    Attributes:
        name: A short, human-readable dataset name.
        source: Where this data came from (e.g. ``"Roark's Formulas for
            Stress and Strain, 8th ed., Table 8.1"``, a lab report
            reference, a published paper citation, or
            ``"self-generated: <formula>"`` for a dataset built from
            this toolkit's own analytical benchmark formulas -- never
            left blank, and never claiming experimental origin unless
            it genuinely has one).
        quantity: The physical quantity the values represent (e.g.
            ``"Tip displacement"``).
        units: The physical units of :attr:`values`.
        independent_variable: The coordinate or independent-variable
            values each entry in :attr:`values` corresponds to (e.g.
            positions along a beam, load magnitudes, mesh sizes).
        independent_variable_label: A label for
            :attr:`independent_variable` (e.g. ``"x (m)"``).
        values: The reference values themselves.
        uncertainty: Optional per-point measurement/reference
            uncertainty, same shape as :attr:`values`. ``None`` if not
            known or not applicable (e.g. for an analytically-derived
            reference, which has no measurement uncertainty).
        description: A longer description of the dataset and how it was
            obtained.

    Raises:
        ValidationError: If :attr:`values` and :attr:`independent_variable`
            (or :attr:`uncertainty`, when given) have different lengths,
            or if :attr:`name`/:attr:`source`/:attr:`quantity` is blank.
    """

    name: str
    source: str
    quantity: str
    units: str
    independent_variable: np.ndarray
    independent_variable_label: str
    values: np.ndarray
    uncertainty: np.ndarray | None = None
    description: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("name", "source", "quantity"):
            if not getattr(self, field_name).strip():
                raise ValidationError(f"ReferenceDataset.{field_name} must not be blank.")
        if len(self.independent_variable) != len(self.values):
            raise ValidationError(
                "ReferenceDataset.independent_variable and .values must have the same "
                f"length, got {len(self.independent_variable)} and {len(self.values)}."
            )
        if self.uncertainty is not None and len(self.uncertainty) != len(self.values):
            raise ValidationError(
                "ReferenceDataset.uncertainty must have the same length as .values, got "
                f"{len(self.uncertainty)} and {len(self.values)}."
            )


def reference_dataset_from_dict(data: dict) -> ReferenceDataset:
    """Build a :class:`ReferenceDataset` from a plain dictionary (e.g. parsed JSON).

    Args:
        data: A dictionary with keys matching :class:`ReferenceDataset`'s
            fields (``independent_variable``/``values``/``uncertainty``
            given as plain lists of numbers).

    Returns:
        The constructed :class:`ReferenceDataset`.

    Raises:
        ValidationError: If a required key is missing, or the resulting
            dataset fails its own validation.
    """
    required = ("name", "source", "quantity", "units", "independent_variable", "values")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValidationError(f"Reference dataset data is missing required key(s): {missing}.")

    uncertainty = data.get("uncertainty")
    return ReferenceDataset(
        name=data["name"],
        source=data["source"],
        quantity=data["quantity"],
        units=data["units"],
        independent_variable=np.asarray(data["independent_variable"], dtype=float),
        independent_variable_label=data.get("independent_variable_label", ""),
        values=np.asarray(data["values"], dtype=float),
        uncertainty=None if uncertainty is None else np.asarray(uncertainty, dtype=float),
        description=data.get("description", ""),
        metadata=dict(data.get("metadata", {})),
    )


def load_reference_dataset_json(path: str | Path) -> ReferenceDataset:
    """Load a :class:`ReferenceDataset` from a JSON file.

    Args:
        path: Path to a JSON file shaped like
            :func:`reference_dataset_from_dict`'s expected input.

    Returns:
        The loaded :class:`ReferenceDataset`.

    Raises:
        ValidationError: If the file does not exist, is not valid JSON,
            or is missing a required field.
    """
    resolved_path = Path(path)
    if not resolved_path.is_file():
        raise ValidationError(f"Reference dataset file not found: {resolved_path}.")
    try:
        data = json.loads(resolved_path.read_text())
    except json.JSONDecodeError as error:
        raise ValidationError(f"Reference dataset file is not valid JSON: {error}.") from error
    return reference_dataset_from_dict(data)


def load_reference_dataset_csv(
    path: str | Path,
    name: str,
    source: str,
    quantity: str,
    units: str,
    independent_variable_label: str = "",
    description: str = "",
) -> ReferenceDataset:
    """Load a :class:`ReferenceDataset` from a two-column CSV file.

    Args:
        path: Path to a CSV file with a header row followed by rows of
            ``independent_variable,value`` (an optional third column,
            ``uncertainty``, is read if present).
        name: The dataset name (CSV files carry no metadata of their
            own, so this and the following arguments supply it).
        source: Where this data came from.
        quantity: The physical quantity the values represent.
        units: The physical units of the values column.
        independent_variable_label: A label for the independent-variable
            column.
        description: A longer description of the dataset.

    Returns:
        The loaded :class:`ReferenceDataset`.

    Raises:
        ValidationError: If the file does not exist, has fewer than two
            columns, or contains a non-numeric value.
    """
    resolved_path = Path(path)
    if not resolved_path.is_file():
        raise ValidationError(f"Reference dataset file not found: {resolved_path}.")

    independent_variable: list[float] = []
    values: list[float] = []
    uncertainty: list[float] = []
    has_uncertainty = False

    with resolved_path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)
        if header is None or len(header) < 2:
            raise ValidationError(
                "Reference dataset CSV must have a header row with at least two columns "
                "(independent_variable, value)."
            )
        has_uncertainty = len(header) >= 3
        for row in reader:
            try:
                independent_variable.append(float(row[0]))
                values.append(float(row[1]))
                if has_uncertainty:
                    uncertainty.append(float(row[2]))
            except ValueError as error:
                raise ValidationError(
                    f"Reference dataset CSV contains a non-numeric value in row {row!r}."
                ) from error

    return ReferenceDataset(
        name=name,
        source=source,
        quantity=quantity,
        units=units,
        independent_variable=np.asarray(independent_variable, dtype=float),
        independent_variable_label=independent_variable_label,
        values=np.asarray(values, dtype=float),
        uncertainty=np.asarray(uncertainty, dtype=float) if has_uncertainty else None,
        description=description,
    )


__all__ = [
    "ReferenceDataset",
    "load_reference_dataset_csv",
    "load_reference_dataset_json",
    "reference_dataset_from_dict",
]
