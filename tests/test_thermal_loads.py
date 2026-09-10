"""Tests for femtoolkit.thermal.thermal_loads: volumetric heat generation."""

import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Node, Tet4Element3D
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.thermal.thermal_loads import (
    HeatGeneration,
    element_heat_generation_to_thermal_loads,
    heat_generation_to_thermal_loads,
)


@pytest.fixture
def tet_mesh() -> tuple[Mesh, Tet4Element3D]:
    mat = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=1.0, y=0.0, z=0.0),
        Node(id=3, x=0.0, y=1.0, z=0.0),
        Node(id=4, x=0.0, y=0.0, z=1.0),
    )
    tet = Tet4Element3D(id=1, nodes=nodes, material=mat)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(tet)
    return mesh, tet


@pytest.fixture
def hex_mesh() -> tuple[Mesh, Hex8Element3D]:
    mat = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    coords = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(coords))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=mat)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)
    return mesh, hexa


def test_zero_generation_gives_zero_loads(tet_mesh: tuple[Mesh, Tet4Element3D]) -> None:
    mesh, _ = tet_mesh
    loads = heat_generation_to_thermal_loads(mesh, HeatGeneration(volumetric_rate=0.0))
    assert all(load.value == pytest.approx(0.0) for load in loads)


def test_uniform_generation_tet4_splits_evenly(tet_mesh: tuple[Mesh, Tet4Element3D]) -> None:
    mesh, tet = tet_mesh
    rate = 1.0e5
    loads = heat_generation_to_thermal_loads(mesh, HeatGeneration(volumetric_rate=rate))
    assert len(loads) == 4
    expected_share = rate * tet.volume / 4.0
    for load in loads:
        assert load.value == pytest.approx(expected_share)
    assert sum(load.value for load in loads) == pytest.approx(rate * tet.volume)


def test_uniform_generation_hex8_totals_correctly(hex_mesh: tuple[Mesh, Hex8Element3D]) -> None:
    mesh, hexa = hex_mesh
    rate = 2.0e4
    loads = heat_generation_to_thermal_loads(mesh, HeatGeneration(volumetric_rate=rate))
    assert len(loads) == 8
    assert sum(load.value for load in loads) == pytest.approx(rate * 1.0, rel=1e-8)


def test_negative_generation_is_a_heat_sink(tet_mesh: tuple[Mesh, Tet4Element3D]) -> None:
    mesh, tet = tet_mesh
    loads = heat_generation_to_thermal_loads(mesh, HeatGeneration(volumetric_rate=-5.0e4))
    assert all(load.value < 0.0 for load in loads)


def test_element_based_heat_generation_only_affects_specified_elements() -> None:
    mat = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes_a = (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=1.0, y=0.0, z=0.0),
        Node(id=3, x=0.0, y=1.0, z=0.0),
        Node(id=4, x=0.0, y=0.0, z=1.0),
    )
    nodes_b = (
        Node(id=5, x=2.0, y=0.0, z=0.0),
        Node(id=6, x=3.0, y=0.0, z=0.0),
        Node(id=7, x=2.0, y=1.0, z=0.0),
        Node(id=8, x=2.0, y=0.0, z=1.0),
    )
    tet_a = Tet4Element3D(id=1, nodes=nodes_a, material=mat)
    tet_b = Tet4Element3D(id=2, nodes=nodes_b, material=mat)
    mesh = Mesh()
    for node in nodes_a + nodes_b:
        mesh.add_node(node)
    mesh.add_element(tet_a)
    mesh.add_element(tet_b)

    loads = element_heat_generation_to_thermal_loads(mesh, {1: 1.0e5})
    loaded_node_ids = {load.node_id for load in loads}
    assert loaded_node_ids == {1, 2, 3, 4}
    assert sum(load.value for load in loads) == pytest.approx(1.0e5 * tet_a.volume)


def test_element_based_heat_generation_skips_elements_not_in_mapping(
    tet_mesh: tuple[Mesh, Tet4Element3D],
) -> None:
    mesh, _ = tet_mesh
    loads = element_heat_generation_to_thermal_loads(mesh, {})
    assert loads == []


def test_heat_generation_rejects_non_finite_rate() -> None:
    with pytest.raises(ValidationError):
        HeatGeneration(volumetric_rate=float("nan"))


def test_contributions_from_shared_nodes_are_not_summed_by_the_converter() -> None:
    """heat_generation_to_thermal_loads returns one ThermalLoad per node per element
    (summed later by the assembler, matching the GravityLoad/TemperatureLoad precedent)."""
    mat = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=1.0, y=0.0, z=0.0),
        Node(id=3, x=0.0, y=1.0, z=0.0),
        Node(id=4, x=0.0, y=0.0, z=1.0),
        Node(id=5, x=1.0, y=1.0, z=1.0),
    )
    tet_a = Tet4Element3D(id=1, nodes=nodes[:4], material=mat)
    tet_b = Tet4Element3D(id=2, nodes=(nodes[1], nodes[2], nodes[3], nodes[4]), material=mat)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(tet_a)
    mesh.add_element(tet_b)

    loads = heat_generation_to_thermal_loads(mesh, HeatGeneration(volumetric_rate=1.0e4))
    node_2_contributions = [load for load in loads if load.node_id == 2]
    assert len(node_2_contributions) == 2  # both elements touch node 2
