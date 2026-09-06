"""Tests for femtoolkit.analysis.temperature_field."""

import pytest

from femtoolkit.analysis.temperature_field import TemperatureField, thermoelastic_materials_for_mesh
from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import LinearElastic3D, ThermoelasticMaterial3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node, Tet4Element3D

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


@pytest.fixture
def placeholder() -> LinearElastic3D:
    return LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)


@pytest.fixture
def hexa(placeholder: LinearElastic3D) -> Hex8Element3D:
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    return Hex8Element3D(id=1, nodes=nodes, material=placeholder)


@pytest.fixture
def tet(placeholder: LinearElastic3D) -> Tet4Element3D:
    nodes = (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=1.0, y=0.0, z=0.0),
        Node(id=3, x=0.0, y=1.0, z=0.0),
        Node(id=4, x=0.0, y=0.0, z=1.0),
    )
    return Tet4Element3D(id=2, nodes=nodes, material=placeholder)


def test_uniform_field_gives_same_temperature_at_every_node() -> None:
    field = TemperatureField.uniform(373.15)
    assert field.node_temperature(1) == pytest.approx(373.15)
    assert field.node_temperature(999) == pytest.approx(373.15)


def test_uniform_field_element_temperature(hexa: Hex8Element3D) -> None:
    field = TemperatureField.uniform(373.15)
    assert field.element_temperature(hexa) == pytest.approx(373.15)


def test_nodal_field_returns_per_node_values() -> None:
    field = TemperatureField.nodal({1: 300.0, 2: 350.0})
    assert field.node_temperature(1) == pytest.approx(300.0)
    assert field.node_temperature(2) == pytest.approx(350.0)


def test_nodal_field_missing_node_raises() -> None:
    field = TemperatureField.nodal({1: 300.0})
    with pytest.raises(ValidationError):
        field.node_temperature(2)


def test_nodal_field_element_temperature_is_average(tet: Tet4Element3D) -> None:
    field = TemperatureField.nodal({1: 300.0, 2: 320.0, 3: 340.0, 4: 360.0})
    assert field.element_temperature(tet) == pytest.approx((300.0 + 320.0 + 340.0 + 360.0) / 4.0)


def test_cannot_construct_directly_without_a_representation() -> None:
    with pytest.raises(ValidationError):
        TemperatureField()


def test_cannot_construct_directly_with_both_representations() -> None:
    with pytest.raises(ValidationError):
        TemperatureField(_uniform_temperature=300.0, _nodal_temperatures={1: 300.0})


def test_rejects_non_finite_uniform_temperature() -> None:
    with pytest.raises(ValidationError):
        TemperatureField.uniform(float("nan"))


def test_rejects_empty_nodal_temperatures() -> None:
    with pytest.raises(ValidationError):
        TemperatureField.nodal({})


def test_rejects_non_finite_nodal_temperature() -> None:
    with pytest.raises(ValidationError):
        TemperatureField.nodal({1: float("inf")})


def test_thermoelastic_materials_for_mesh_builds_per_element_materials(
    hexa: Hex8Element3D, placeholder: LinearElastic3D
) -> None:
    mesh = Mesh()
    for node in hexa.nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    base_material = ThermoelasticMaterial3D(
        youngs_modulus=200e9,
        poisson_ratio=0.3,
        thermal_expansion_coefficient=12e-6,
        reference_temperature=293.15,
        density=7850.0,
    )
    field = TemperatureField.uniform(393.15)
    materials = thermoelastic_materials_for_mesh(mesh, base_material, field)

    assert set(materials) == {hexa.id}
    assert materials[hexa.id].temperature == pytest.approx(393.15)


def test_thermoelastic_materials_for_mesh_gives_different_materials_for_different_temperatures(
    placeholder: LinearElastic3D,
) -> None:
    nodes_a = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    element_a = Hex8Element3D(id=1, nodes=nodes_a, material=placeholder)
    nodes_b = tuple(
        Node(id=i + 9, x=x + 1.0, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS)
    )
    element_b = Hex8Element3D(id=2, nodes=nodes_b, material=placeholder)

    mesh = Mesh()
    for node in nodes_a + nodes_b:
        mesh.add_node(node)
    mesh.add_element(element_a)
    mesh.add_element(element_b)

    base_material = ThermoelasticMaterial3D(
        youngs_modulus=200e9,
        poisson_ratio=0.3,
        thermal_expansion_coefficient=12e-6,
        reference_temperature=293.15,
        density=7850.0,
    )
    nodal_temperatures = {node.id: 293.15 for node in nodes_a}
    nodal_temperatures.update({node.id: 393.15 for node in nodes_b})
    field = TemperatureField.nodal(nodal_temperatures)
    materials = thermoelastic_materials_for_mesh(mesh, base_material, field)

    assert materials[element_a.id].temperature == pytest.approx(293.15)
    assert materials[element_b.id].temperature == pytest.approx(393.15)
