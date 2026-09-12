"""Tests for femtoolkit.thermal.thermal_surfaces."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis.dof import DOFMap
from femtoolkit.exceptions import EntityNotFoundError, ValidationError
from femtoolkit.materials import LinearElastic2D, LinearElastic3D, Material
from femtoolkit.mesh import Hex8Element3D, Mesh, Node, Tet4Element3D
from femtoolkit.mesh.bar_element import BarElement
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.quad_element import QuadElement2D
from femtoolkit.sections import CrossSection
from femtoolkit.thermal.thermal_surfaces import (
    ThermalSurface,
    surface_area,
    surface_conductance_contribution,
    surface_mean_temperature,
    surface_node_ids,
    surface_uniform_load,
)

_HEX8_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def _hex8_mesh() -> tuple[Mesh, Hex8Element3D]:
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_HEX8_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)
    return mesh, hexa


def _tet4_mesh() -> tuple[Mesh, Tet4Element3D]:
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=1.0, y=0.0, z=0.0),
        Node(id=3, x=0.0, y=1.0, z=0.0),
        Node(id=4, x=0.0, y=0.0, z=1.0),
    )
    tet = Tet4Element3D(id=1, nodes=nodes, material=placeholder)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(tet)
    return mesh, tet


def _cst_mesh() -> tuple[Mesh, CSTElement2D]:
    placeholder = LinearElastic2D(
        youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress"
    )
    nodes = (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=1.0, y=0.0, z=0.0),
        Node(id=3, x=0.0, y=1.0, z=0.0),
    )
    cst = CSTElement2D(id=1, nodes=nodes, material=placeholder, thickness=0.1)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(cst)
    return mesh, cst


def _quad_mesh() -> tuple[Mesh, QuadElement2D]:
    placeholder = LinearElastic2D(
        youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress"
    )
    nodes = (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=2.0, y=0.0, z=0.0),
        Node(id=3, x=2.0, y=1.0, z=0.0),
        Node(id=4, x=0.0, y=1.0, z=0.0),
    )
    quad = QuadElement2D(id=1, nodes=nodes, material=placeholder, thickness=0.5)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(quad)
    return mesh, quad


def test_thermal_surface_rejects_non_positive_element_id() -> None:
    with pytest.raises(ValidationError):
        ThermalSurface(element_id=0, local_face_index=0)


def test_thermal_surface_rejects_negative_face_index() -> None:
    with pytest.raises(ValidationError):
        ThermalSurface(element_id=1, local_face_index=-1)


def test_cst_edge_node_ids_and_areas() -> None:
    mesh, _cst = _cst_mesh()
    assert surface_node_ids(mesh, ThermalSurface(1, 0)) == (1, 2)
    assert surface_node_ids(mesh, ThermalSurface(1, 1)) == (2, 3)
    assert surface_node_ids(mesh, ThermalSurface(1, 2)) == (3, 1)
    assert surface_area(mesh, ThermalSurface(1, 0)) == pytest.approx(1.0 * 0.1)
    assert surface_area(mesh, ThermalSurface(1, 2)) == pytest.approx(1.0 * 0.1)
    assert surface_area(mesh, ThermalSurface(1, 1)) == pytest.approx(2.0**0.5 * 0.1)


def test_quad_edge_node_ids_and_areas() -> None:
    mesh, _quad = _quad_mesh()
    assert surface_node_ids(mesh, ThermalSurface(1, 0)) == (1, 2)
    assert surface_area(mesh, ThermalSurface(1, 0)) == pytest.approx(2.0 * 0.5)
    assert surface_area(mesh, ThermalSurface(1, 1)) == pytest.approx(1.0 * 0.5)


def test_tet4_face_node_ids_and_areas() -> None:
    mesh, _tet = _tet4_mesh()
    surf = ThermalSurface(1, 3)  # (1,2,3) -> the hypotenuse face
    assert set(surface_node_ids(mesh, surf)) == {2, 3, 4}
    assert surface_area(mesh, surf) == pytest.approx(3.0**0.5 / 2.0)
    assert surface_area(mesh, ThermalSurface(1, 0)) == pytest.approx(0.5)


def test_hex8_face_node_ids_and_areas() -> None:
    mesh, _hexa = _hex8_mesh()
    for face_index in range(6):
        assert surface_area(mesh, ThermalSurface(1, face_index)) == pytest.approx(1.0)
    assert surface_node_ids(mesh, ThermalSurface(1, 0)) == (1, 2, 3, 4)
    assert surface_node_ids(mesh, ThermalSurface(1, 1)) == (5, 6, 7, 8)


def test_surface_conductance_contribution_cst_edge() -> None:
    mesh, _cst = _cst_mesh()
    contribution = surface_conductance_contribution(mesh, ThermalSurface(1, 0), 25.0)
    assert contribution.stiffness.shape == (2, 2)
    expected_area = surface_area(mesh, ThermalSurface(1, 0))
    assert contribution.stiffness.sum() == pytest.approx(25.0 * expected_area)
    node_ids = {key[0] for key in contribution.dof_keys}
    assert node_ids == {1, 2}


def test_surface_conductance_contribution_hex8_face() -> None:
    mesh, _hexa = _hex8_mesh()
    contribution = surface_conductance_contribution(mesh, ThermalSurface(1, 0), 5.0)
    assert contribution.stiffness.shape == (4, 4)
    assert contribution.stiffness.sum() == pytest.approx(5.0)


def test_surface_uniform_load_cst_edge_splits_evenly() -> None:
    mesh, _cst = _cst_mesh()
    loads = surface_uniform_load(mesh, ThermalSurface(1, 0), 100.0)
    assert len(loads) == 2
    total = sum(load.value for load in loads)
    assert total == pytest.approx(100.0 * surface_area(mesh, ThermalSurface(1, 0)))
    assert_allclose([load.value for load in loads], total / 2.0)


def test_surface_uniform_load_tet4_face_splits_evenly() -> None:
    mesh, _tet = _tet4_mesh()
    surf = ThermalSurface(1, 0)
    loads = surface_uniform_load(mesh, surf, 60.0)
    total = sum(load.value for load in loads)
    assert total == pytest.approx(60.0 * surface_area(mesh, surf))
    assert_allclose([load.value for load in loads], total / 3.0)


def test_surface_mean_temperature() -> None:
    mesh, _cst = _cst_mesh()
    dof_map = DOFMap(node_ids=[node.id for node in mesh.nodes], dofs_per_node=1)
    temperatures = np.zeros(dof_map.total_dofs)
    temperatures[dof_map.global_index(1, 0)] = 300.0
    temperatures[dof_map.global_index(2, 0)] = 320.0
    mean = surface_mean_temperature(mesh, ThermalSurface(1, 0), temperatures, dof_map)
    assert mean == pytest.approx(310.0)


def test_surface_area_rejects_out_of_range_face_index() -> None:
    mesh, _hexa = _hex8_mesh()
    with pytest.raises(ValidationError):
        surface_area(mesh, ThermalSurface(1, 6))


def test_surface_area_rejects_missing_element() -> None:
    mesh, _hexa = _hex8_mesh()
    with pytest.raises(EntityNotFoundError):
        surface_area(mesh, ThermalSurface(999, 0))


def test_surface_area_rejects_bar_element() -> None:
    material = Material(name="steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    n1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    n2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    bar = BarElement(id=1, nodes=(n1, n2), material=material, cross_section=CrossSection(area=0.01))
    mesh = Mesh()
    mesh.add_node(n1)
    mesh.add_node(n2)
    mesh.add_element(bar)
    with pytest.raises(ValidationError):
        surface_area(mesh, ThermalSurface(1, 0))
