"""Writing a :class:`~femtoolkit.postprocessing.result_model.SimulationResult` to a file.

Two formats are supported, and the dispatch (:func:`export_result`) is
written so a third can be added later without touching the two existing
writers:

* **CSV** (:func:`export_to_csv`) -- one *step* at a time (defaulting to
  the last/converged step), in **tidy** (long) form: one row per
  ``(entity, field, component)`` triple, rather than one column per
  field. A tidy table avoids the column-count mismatch a "wide" table
  would hit the moment two fields have different numbers of components
  (a scalar temperature next to a 3-component displacement, for
  example), and is directly importable into a spreadsheet or
  ``pandas.read_csv`` without any special handling.
* **JSON** (:func:`export_to_json`) -- the *entire* result (every step),
  since JSON's nested structure has no equivalent "one row per
  observation" constraint -- a single file naturally holds the full
  topology plus every step's fields.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Literal

import numpy as np

from femtoolkit.exceptions import ValidationError
from femtoolkit.postprocessing.result_model import FieldValue, SimulationResult

ExportFormat = Literal["csv", "json"]


def _field_rows(field_name: str, values: dict[int, FieldValue]) -> list[tuple[int, str, float]]:
    """Flatten one field's ``{entity_id: value}`` mapping into
    ``(entity_id, component, value)`` rows."""
    rows: list[tuple[int, str, float]] = []
    for entity_id, value in values.items():
        array = np.atleast_1d(np.asarray(value, dtype=float))
        if array.size == 1:
            rows.append((entity_id, "", float(array[0])))
        else:
            for component_index, component_value in enumerate(array):
                rows.append((entity_id, str(component_index), float(component_value)))
    return rows


def export_to_csv(result: SimulationResult, path: str | Path, step: int = -1) -> None:
    """Write one step's nodal and element fields to a tidy CSV file.

    Each row is one ``(entity, field, component)`` observation. Node
    rows additionally carry the node's original coordinates;
    element rows carry their connectivity (semicolon-joined node IDs).

    Args:
        result: The result to export.
        path: Destination file path.
        step: Which step to export (default: the last/converged step).
    """
    result_step = result.step(step)
    topology = result.topology

    with Path(path).open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "entity_type",
                "entity_id",
                "x",
                "y",
                "z",
                "connectivity",
                "field",
                "component",
                "value",
                "unit",
                "step",
                "time",
            ]
        )

        for field_name, values in result_step.nodal_fields.items():
            unit = result.field_units.get(field_name, "")
            for node_id, component, value in _field_rows(field_name, values):
                x, y, z = topology.node_coordinates[node_id]
                writer.writerow(
                    ["node", node_id, x, y, z, "", field_name, component, value, unit,
                     result_step.index, result_step.time]
                )

        for field_name, values in result_step.element_fields.items():
            unit = result.field_units.get(field_name, "")
            for element_id, component, value in _field_rows(field_name, values):
                connectivity = ";".join(str(n) for n in topology.element_connectivity[element_id])
                writer.writerow(
                    ["element", element_id, "", "", "", connectivity, field_name, component,
                     value, unit, result_step.index, result_step.time]
                )


def _to_jsonable(value: object) -> object:
    """Recursively convert NumPy scalars/arrays (and nested containers of them) to plain Python
    types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def export_to_json(result: SimulationResult, path: str | Path) -> None:
    """Write the entire result (topology and every step) to a JSON file.

    Args:
        result: The result to export.
        path: Destination file path.
    """
    payload = {
        "topology": {
            "node_ids": list(result.topology.node_ids),
            "node_coordinates": result.topology.node_coordinates,
            "element_ids": list(result.topology.element_ids),
            "element_connectivity": result.topology.element_connectivity,
            "element_types": result.topology.element_types,
        },
        "field_units": result.field_units,
        "steps": [
            {
                "index": step.index,
                "time": step.time,
                "nodal_fields": step.nodal_fields,
                "element_fields": step.element_fields,
            }
            for step in result.steps
        ],
    }
    with Path(path).open("w", encoding="utf-8") as json_file:
        json.dump(_to_jsonable(payload), json_file, indent=2)


def export_result(result: SimulationResult, path: str | Path, export_format: ExportFormat) -> None:
    """Dispatch to :func:`export_to_csv` or :func:`export_to_json` by format name.

    Args:
        result: The result to export.
        path: Destination file path.
        export_format: ``"csv"`` or ``"json"``.

    Raises:
        ValidationError: If ``export_format`` is not a supported format.
    """
    if export_format == "csv":
        export_to_csv(result, path)
        return
    if export_format == "json":
        export_to_json(result, path)
        return
    raise ValidationError(f'export_format must be "csv" or "json", got {export_format!r}.')
