"""Tests for femtoolkit.postprocessing.visualization_3d.mesh_converter."""

import numpy as np
import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import LinearElastic2D, LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node, Tet4Element3D
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.quad_element import QuadElement2D
from femtoolkit.postprocessing.result_model import MeshTopology
from femtoolkit.postprocessing.visualization_3d.mesh_converter import (
    attach_element_field,
    attach_nodal_field,
    mesh_topology_to_grid,
    result_step_to_grid,
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


def test_tet4_conversion_has_correct_node_count_cell_count_and_volume() -> None:
    mesh, _tet = _tet4_mesh()
    topology = MeshTopology.from_mesh(mesh)
    grid = mesh_topology_to_grid(topology)

    assert grid.n_points == 4
    assert grid.n_cells == 1
    assert grid.volume == pytest.approx(1.0 / 6.0)


def test_hex8_conversion_has_correct_node_count_cell_count_and_volume() -> None:
    mesh, _hexa = _hex8_mesh()
    topology = MeshTopology.from_mesh(mesh)
    grid = mesh_topology_to_grid(topology)

    assert grid.n_points == 8
    assert grid.n_cells == 1
    assert grid.volume == pytest.approx(1.0)


def test_hex8_conversion_preserves_node_coordinates() -> None:
    mesh, _hexa = _hex8_mesh()
    topology = MeshTopology.from_mesh(mesh)
    grid = mesh_topology_to_grid(topology)

    for index, node_id in enumerate(topology.node_ids):
        assert tuple(grid.points[index]) == pytest.approx(topology.node_coordinates[node_id])


def test_hex8_conversion_preserves_connectivity_order() -> None:
    mesh, hexa = _hex8_mesh()
    topology = MeshTopology.from_mesh(mesh)
    grid = mesh_topology_to_grid(topology)

    cell_point_ids = grid.get_cell(0).point_ids
    node_index = {node_id: index for index, node_id in enumerate(topology.node_ids)}
    expected = [node_index[node.id] for node in hexa.nodes]
    assert cell_point_ids == expected


def test_cst_conversion_has_correct_area() -> None:
    mesh, _cst = _cst_mesh()
    topology = MeshTopology.from_mesh(mesh)
    grid = mesh_topology_to_grid(topology)

    assert grid.n_points == 3
    assert grid.n_cells == 1
    assert grid.area == pytest.approx(0.5)


def test_quad_conversion_has_correct_area() -> None:
    mesh, _quad = _quad_mesh()
    topology = MeshTopology.from_mesh(mesh)
    grid = mesh_topology_to_grid(topology)

    assert grid.n_points == 4
    assert grid.n_cells == 1
    assert grid.area == pytest.approx(2.0)


def test_unsupported_element_type_raises_validation_error() -> None:
    topology = MeshTopology(
        node_ids=(1, 2),
        node_coordinates={1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0)},
        element_ids=(1,),
        element_connectivity={1: (1, 2)},
        element_types={1: "FrameElement2D"},
    )
    with pytest.raises(ValidationError):
        mesh_topology_to_grid(topology)


def test_attach_nodal_scalar_field() -> None:
    mesh, hexa = _hex8_mesh()
    topology = MeshTopology.from_mesh(mesh)
    grid = mesh_topology_to_grid(topology)
    values = {node.id: 300.0 + node.id for node in hexa.nodes}
    attach_nodal_field(grid, topology, "temperature", values)

    for index, node_id in enumerate(topology.node_ids):
        assert grid.point_data["temperature"][index] == pytest.approx(values[node_id])


def test_attach_nodal_field_fills_missing_nodes_with_nan() -> None:
    mesh, hexa = _hex8_mesh()
    topology = MeshTopology.from_mesh(mesh)
    grid = mesh_topology_to_grid(topology)
    partial_values = {hexa.nodes[0].id: 400.0}
    attach_nodal_field(grid, topology, "temperature", partial_values)

    array = grid.point_data["temperature"]
    assert array[0] == pytest.approx(400.0)
    assert np.isnan(array[1])


def test_attach_element_vector_field_pads_to_three_components() -> None:
    mesh, _cst = _cst_mesh()
    topology = MeshTopology.from_mesh(mesh)
    grid = mesh_topology_to_grid(topology)
    attach_element_field(grid, topology, "heat_flux", {1: np.array([10.0, 20.0])})

    array = grid.cell_data["heat_flux"]
    assert array.shape == (1, 3)
    assert array[0] == pytest.approx([10.0, 20.0, 0.0])


def test_result_step_to_grid_attaches_every_field() -> None:
    mesh, hexa = _hex8_mesh()
    topology = MeshTopology.from_mesh(mesh)
    from femtoolkit.postprocessing.result_model import ResultStep

    step = ResultStep(
        index=0,
        time=0.0,
        nodal_fields={"temperature": {node.id: 300.0 for node in hexa.nodes}},
        element_fields={"heat_flux": {hexa.id: np.array([1.0, 0.0, 0.0])}},
    )
    grid = result_step_to_grid(topology, step)

    assert "temperature" in grid.point_data
    assert "heat_flux" in grid.cell_data
