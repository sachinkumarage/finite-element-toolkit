"""Tests for femtoolkit.postprocessing.export."""

import csv
import json

import numpy as np
import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.postprocessing.export import export_result, export_to_csv, export_to_json
from femtoolkit.postprocessing.result_model import MeshTopology, ResultStep, SimulationResult

_HEX8_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def _topology() -> MeshTopology:
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_HEX8_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)
    return MeshTopology.from_mesh(mesh)


def _result() -> SimulationResult:
    topology = _topology()
    step0 = ResultStep(
        index=0,
        time=0.0,
        nodal_fields={"temperature": {1: 300.0, 2: 350.0}},
        element_fields={"heat_flux": {1: np.array([100.0, 0.0, 0.0])}},
    )
    step1 = ResultStep(
        index=1,
        time=1.0,
        nodal_fields={"temperature": {1: 300.0, 2: 400.0}},
        element_fields={"heat_flux": {1: np.array([120.0, 0.0, 0.0])}},
    )
    return SimulationResult(topology, steps=(step0, step1), field_units={"temperature": "K"})


def test_export_to_csv_writes_expected_header_and_rows(tmp_path) -> None:
    result = _result()
    path = tmp_path / "result.csv"
    export_to_csv(result, path, step=-1)

    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert rows
    assert set(rows[0]) == {
        "entity_type", "entity_id", "x", "y", "z", "connectivity",
        "field", "component", "value", "unit", "step", "time",
    }

    temperature_rows = [row for row in rows if row["field"] == "temperature"]
    node_2_row = next(row for row in temperature_rows if row["entity_id"] == "2")
    assert float(node_2_row["value"]) == pytest.approx(400.0)
    assert node_2_row["unit"] == "K"
    assert node_2_row["step"] == "1"

    flux_rows = [row for row in rows if row["field"] == "heat_flux"]
    assert len(flux_rows) == 3  # 3 components of the heat flux vector
    x_component = next(row for row in flux_rows if row["component"] == "0")
    assert float(x_component["value"]) == pytest.approx(120.0)
    assert x_component["connectivity"] != ""


def test_export_to_csv_exports_requested_step_not_only_the_last(tmp_path) -> None:
    result = _result()
    path = tmp_path / "result_step0.csv"
    export_to_csv(result, path, step=0)

    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    node_2_row = next(
        row for row in rows if row["field"] == "temperature" and row["entity_id"] == "2"
    )
    assert float(node_2_row["value"]) == pytest.approx(350.0)


def test_export_to_json_includes_topology_and_every_step(tmp_path) -> None:
    result = _result()
    path = tmp_path / "result.json"
    export_to_json(result, path)

    with path.open(encoding="utf-8") as json_file:
        payload = json.load(json_file)

    assert set(payload["topology"]["node_ids"]) == {1, 2, 3, 4, 5, 6, 7, 8}
    assert payload["topology"]["node_coordinates"]["1"] == [0.0, 0.0, 0.0]
    assert len(payload["steps"]) == 2
    assert payload["steps"][1]["nodal_fields"]["temperature"]["2"] == pytest.approx(400.0)
    assert payload["steps"][0]["element_fields"]["heat_flux"]["1"] == pytest.approx(
        [100.0, 0.0, 0.0]
    )
    assert payload["field_units"]["temperature"] == "K"


def test_export_to_json_values_are_plain_python_types_not_numpy(tmp_path) -> None:
    result = _result()
    path = tmp_path / "result.json"
    export_to_json(result, path)
    # json.load succeeding at all proves every stored value serialized cleanly;
    # this additionally checks the raw text contains no numpy repr artifacts.
    raw_text = path.read_text(encoding="utf-8")
    assert "array(" not in raw_text
    assert "np.float64" not in raw_text


def test_export_result_dispatches_csv(tmp_path) -> None:
    result = _result()
    path = tmp_path / "result.csv"
    export_result(result, path, "csv")
    assert path.exists()


def test_export_result_dispatches_json(tmp_path) -> None:
    result = _result()
    path = tmp_path / "result.json"
    export_result(result, path, "json")
    assert path.exists()


def test_export_result_rejects_unsupported_format(tmp_path) -> None:
    result = _result()
    with pytest.raises(ValidationError):
        export_result(result, tmp_path / "result.xml", "xml")
