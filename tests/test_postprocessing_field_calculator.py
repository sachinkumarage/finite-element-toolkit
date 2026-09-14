"""Tests for femtoolkit.postprocessing.field_calculator."""

import numpy as np
import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.postprocessing.field_calculator import (
    deformed_coordinates,
    equivalent_strain,
    equivalent_stress,
    summarize,
    vector_magnitude,
    with_derived_fields,
)
from femtoolkit.postprocessing.result_model import MeshTopology, ResultStep, SimulationResult

_HEX8_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def _hex8_topology() -> MeshTopology:
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_HEX8_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)
    return MeshTopology.from_mesh(mesh)


def test_equivalent_stress_plane_stress_dispatch() -> None:
    # Uniaxial plane-stress state: sigma_vm = |sigma_x|.
    assert equivalent_stress(np.array([100e6, 0.0, 0.0])) == pytest.approx(100e6)


def test_equivalent_stress_3d_dispatch() -> None:
    # Uniaxial 3D state: sigma_vm = |sigma_xx|.
    stress = np.array([100e6, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert equivalent_stress(stress) == pytest.approx(100e6)


def test_equivalent_stress_rejects_unsupported_length() -> None:
    with pytest.raises(ValidationError):
        equivalent_stress(np.array([1.0, 2.0]))


def test_equivalent_strain_uniaxial_incompressible() -> None:
    e0 = 0.01
    strain = np.array([e0, -0.5 * e0, -0.5 * e0, 0.0, 0.0, 0.0])
    assert equivalent_strain(strain) == pytest.approx(e0)


def test_equivalent_strain_rejects_non_6_component() -> None:
    with pytest.raises(ValidationError):
        equivalent_strain(np.array([0.01, 0.0, 0.0]))


def test_vector_magnitude() -> None:
    assert vector_magnitude(np.array([3.0, 4.0])) == pytest.approx(5.0)
    assert vector_magnitude(5.0) == pytest.approx(5.0)


def test_deformed_coordinates_zero_displacement_matches_original() -> None:
    topology = _hex8_topology()
    displacement_field = {node_id: np.zeros(3) for node_id in topology.node_ids}
    deformed = deformed_coordinates(topology, displacement_field, scale=1.0)
    for node_id in topology.node_ids:
        assert deformed[node_id] == topology.node_coordinates[node_id]


def test_deformed_coordinates_known_displacement() -> None:
    topology = _hex8_topology()
    displacement_field = {1: np.array([0.1, 0.0, 0.0])}
    deformed = deformed_coordinates(topology, displacement_field, scale=1.0)
    assert deformed[1] == pytest.approx((0.1, 0.0, 0.0))


def test_deformed_coordinates_scale_factor_is_purely_cosmetic() -> None:
    topology = _hex8_topology()
    displacement_field = {1: np.array([0.01, 0.0, 0.0])}
    unscaled = deformed_coordinates(topology, displacement_field, scale=1.0)
    scaled = deformed_coordinates(topology, displacement_field, scale=100.0)
    assert scaled[1][0] == pytest.approx(100.0 * unscaled[1][0])


def test_deformed_coordinates_handles_2d_displacement_vector() -> None:
    topology = _hex8_topology()
    displacement_field = {1: np.array([0.1, 0.2])}
    deformed = deformed_coordinates(topology, displacement_field, scale=1.0)
    assert deformed[1] == pytest.approx((0.1, 0.2, 0.0))


def test_with_derived_fields_adds_magnitude_for_vector_fields() -> None:
    topology = _hex8_topology()
    step = ResultStep(
        index=0,
        time=0.0,
        element_fields={"heat_flux": {1: np.array([3.0, 4.0, 0.0])}},
    )
    result = SimulationResult(topology, steps=(step,))
    enriched = with_derived_fields(result)
    assert enriched.final_step.element_value("heat_flux_magnitude", 1) == pytest.approx(5.0)


def test_with_derived_fields_adds_von_mises_stress_for_3d_stress() -> None:
    topology = _hex8_topology()
    step = ResultStep(
        index=0,
        time=0.0,
        element_fields={"stress": {1: np.array([100e6, 0.0, 0.0, 0.0, 0.0, 0.0])}},
    )
    result = SimulationResult(topology, steps=(step,))
    enriched = with_derived_fields(result)
    assert enriched.final_step.element_value("von_mises_stress", 1) == pytest.approx(100e6)


def test_with_derived_fields_adds_equivalent_strain_for_3d_strain() -> None:
    topology = _hex8_topology()
    e0 = 0.02
    step = ResultStep(
        index=0,
        time=0.0,
        element_fields={"strain": {1: np.array([e0, -0.5 * e0, -0.5 * e0, 0.0, 0.0, 0.0])}},
    )
    result = SimulationResult(topology, steps=(step,))
    enriched = with_derived_fields(result)
    assert enriched.final_step.element_value("equivalent_strain", 1) == pytest.approx(e0)


def test_with_derived_fields_skips_equivalent_strain_for_2d_strain() -> None:
    topology = _hex8_topology()
    step = ResultStep(
        index=0,
        time=0.0,
        element_fields={"strain": {1: np.array([0.01, -0.005, 0.002])}},
    )
    result = SimulationResult(topology, steps=(step,))
    enriched = with_derived_fields(result)
    assert "equivalent_strain" not in enriched.final_step.element_fields
    # But von Mises stress-style dispatch still works for a 3-component "stress".
    assert "von_mises_stress" not in enriched.final_step.element_fields  # no "stress" field here


def test_with_derived_fields_leaves_original_fields_untouched() -> None:
    topology = _hex8_topology()
    temperatures = {node_id: 300.0 for node_id in topology.node_ids}
    step = ResultStep(index=0, time=0.0, nodal_fields={"temperature": temperatures})
    result = SimulationResult(topology, steps=(step,))
    enriched = with_derived_fields(result)
    assert enriched.final_step.nodal_field("temperature") == step.nodal_field("temperature")


def test_summarize_thermal_only_result_leaves_mechanical_fields_none() -> None:
    topology = _hex8_topology()
    step = ResultStep(
        index=0,
        time=0.0,
        nodal_fields={"temperature": {1: 300.0, 2: 400.0}},
        element_fields={"heat_flux": {1: np.array([100.0, 0.0, 0.0])}},
    )
    result = SimulationResult(topology, steps=(step,))
    summary = summarize(result)

    assert summary.minimum_temperature == pytest.approx(300.0)
    assert summary.maximum_temperature == pytest.approx(400.0)
    assert summary.maximum_heat_flux == pytest.approx(100.0)
    assert summary.maximum_displacement is None
    assert summary.maximum_von_mises_stress is None
    assert summary.num_steps == 1


def test_summarize_mechanical_result_reports_stress_and_strain() -> None:
    topology = _hex8_topology()
    step = ResultStep(
        index=0,
        time=1.0,
        nodal_fields={"displacement": {1: np.array([0.01, 0.0, 0.0])}},
        element_fields={
            "stress": {1: np.array([100e6, 0.0, 0.0, 0.0, 0.0, 0.0])},
            "strain": {1: np.array([0.02, -0.01, -0.01, 0.0, 0.0, 0.0])},
        },
    )
    result = SimulationResult(topology, steps=(step,))
    summary = summarize(result)

    assert summary.maximum_displacement == pytest.approx(0.01)
    assert summary.maximum_von_mises_stress == pytest.approx(100e6)
    assert summary.maximum_equivalent_strain == pytest.approx(0.02)
    assert summary.final_time == pytest.approx(1.0)
