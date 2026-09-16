"""Tests for femtoolkit.postprocessing.visualization_3d.boundary_conditions."""

import pytest

from femtoolkit.analysis.boundary_conditions import BoundaryCondition
from femtoolkit.analysis.dof import TranslationDOF
from femtoolkit.analysis.loads import NodalLoad
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, Node
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.postprocessing.result_model import MeshTopology
from femtoolkit.postprocessing.visualization_3d.boundary_conditions import (
    node_markers,
    surface_markers,
)
from femtoolkit.thermal.thermal_boundary_conditions import (
    PrescribedHeatFlux,
    PrescribedTemperature,
)
from femtoolkit.thermal.thermal_surfaces import ThermalSurface


def _cst_mesh() -> tuple[Mesh, CSTElement2D]:
    material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    nodes = (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=1.0, y=0.0, z=0.0),
        Node(id=3, x=0.0, y=1.0, z=0.0),
    )
    cst = CSTElement2D(id=1, nodes=nodes, material=material, thickness=0.1)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(cst)
    return mesh, cst


def test_node_markers_for_prescribed_temperature() -> None:
    mesh, _cst = _cst_mesh()
    topology = MeshTopology.from_mesh(mesh)
    bcs = [
        PrescribedTemperature(node_id=1, value=373.15),
        PrescribedTemperature(node_id=2, value=300.0),
    ]

    markers = node_markers(topology, [bc.node_id for bc in bcs])

    assert markers.n_points == 2
    assert tuple(markers.points[0]) == pytest.approx((0.0, 0.0, 0.0))
    assert tuple(markers.points[1]) == pytest.approx((1.0, 0.0, 0.0))


def test_node_markers_for_prescribed_heat_flux() -> None:
    mesh, _cst = _cst_mesh()
    topology = MeshTopology.from_mesh(mesh)
    bcs = [PrescribedHeatFlux(node_id=3, value=50.0)]

    markers = node_markers(topology, [bc.node_id for bc in bcs])

    assert markers.n_points == 1
    assert tuple(markers.points[0]) == pytest.approx((0.0, 1.0, 0.0))


def test_node_markers_for_mechanical_constraints() -> None:
    mesh, _cst = _cst_mesh()
    topology = MeshTopology.from_mesh(mesh)
    bcs = [
        BoundaryCondition(node_id=1, dof=TranslationDOF.X, value=0.0),
        BoundaryCondition(node_id=1, dof=TranslationDOF.Y, value=0.0),
    ]

    markers = node_markers(topology, [bc.node_id for bc in bcs])

    assert markers.n_points == 2
    for point in markers.points:
        assert tuple(point) == pytest.approx((0.0, 0.0, 0.0))


def test_node_markers_for_nodal_loads() -> None:
    mesh, _cst = _cst_mesh()
    topology = MeshTopology.from_mesh(mesh)
    loads = [NodalLoad(node_id=2, dof=TranslationDOF.Y, value=-1000.0)]

    markers = node_markers(topology, [load.node_id for load in loads])

    assert markers.n_points == 1
    assert tuple(markers.points[0]) == pytest.approx((1.0, 0.0, 0.0))


def test_node_markers_empty_input() -> None:
    mesh, _cst = _cst_mesh()
    topology = MeshTopology.from_mesh(mesh)

    markers = node_markers(topology, [])

    assert markers.n_points == 0


def test_surface_markers_for_convection_surface() -> None:
    mesh, _cst = _cst_mesh()
    surfaces = [ThermalSurface(element_id=1, local_face_index=0)]

    markers = surface_markers(mesh, surfaces)

    assert markers.n_points == 2
    assert markers.n_cells == 1


def test_surface_markers_multiple_surfaces() -> None:
    mesh, _cst = _cst_mesh()
    surfaces = [
        ThermalSurface(element_id=1, local_face_index=0),
        ThermalSurface(element_id=1, local_face_index=1),
    ]

    markers = surface_markers(mesh, surfaces)

    assert markers.n_points == 4
    assert markers.n_cells == 2


def test_surface_markers_empty_input() -> None:
    mesh, _cst = _cst_mesh()

    markers = surface_markers(mesh, [])

    assert markers.n_points == 0
