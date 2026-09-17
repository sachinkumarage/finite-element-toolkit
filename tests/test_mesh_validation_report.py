"""Tests for femtoolkit.mesh.validation.report (Version 25)."""

import pytest

from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, Node
from femtoolkit.mesh.generator import create_quad_mesh
from femtoolkit.mesh.validation.report import (
    STATUS_ERROR,
    STATUS_OK,
    STATUS_WARNING,
    format_report,
    generate_validation_report,
)


@pytest.fixture
def material() -> LinearElastic2D:
    return LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")


def test_valid_mesh_reports_ok(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01)
    report = generate_validation_report(mesh)

    assert report.status == STATUS_OK
    assert report.is_valid
    assert report.num_duplicate_nodes == 0
    assert report.num_isolated_nodes == 0
    assert report.num_invalid_connectivity == 0
    assert report.num_degenerate_elements == 0
    assert report.num_duplicate_elements == 0
    assert report.warnings == []
    assert report.errors == []
    assert report.quality is not None


def test_isolated_node_reports_warning(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=2, ny=2, material=material, thickness=0.01)
    mesh.add_node(Node(id=9999, x=100.0, y=100.0, z=0.0))

    report = generate_validation_report(mesh)
    assert report.status == STATUS_WARNING
    assert report.is_valid
    assert report.num_isolated_nodes == 1
    assert any("isolated" in warning for warning in report.warnings)


def test_duplicate_node_coordinates_reports_error(material: LinearElastic2D) -> None:
    mesh = Mesh()
    n1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    n2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    n3 = Node(id=3, x=0.0, y=0.0, z=0.0)  # duplicate of n1
    for node in (n1, n2, n3):
        mesh.add_node(node)

    report = generate_validation_report(mesh)
    assert report.status == STATUS_ERROR
    assert not report.is_valid
    assert report.num_duplicate_nodes == 1
    assert report.errors


def test_duplicate_elements_reports_warning(material: LinearElastic2D) -> None:
    from femtoolkit.mesh.cst_element import CSTElement2D

    nodes = (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=1.0, y=0.0, z=0.0),
        Node(id=3, x=0.0, y=1.0, z=0.0),
    )
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(CSTElement2D(id=1, nodes=nodes, material=material, thickness=0.01))
    mesh.add_element(CSTElement2D(id=2, nodes=nodes, material=material, thickness=0.01))

    report = generate_validation_report(mesh)
    assert report.status == STATUS_WARNING
    assert report.num_duplicate_elements == 1


def test_quality_warnings_propagate_into_report(material: LinearElastic2D) -> None:
    from femtoolkit.mesh.quad_element import QuadElement2D

    nodes = (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=10.0, y=0.0, z=0.0),
        Node(id=3, x=10.0, y=0.1, z=0.0),
        Node(id=4, x=0.0, y=0.1, z=0.0),
    )
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(QuadElement2D(id=1, nodes=nodes, material=material, thickness=0.01))

    report = generate_validation_report(mesh, poor_quality_threshold=0.5)
    assert report.status == STATUS_WARNING
    assert any("poor quality" in warning for warning in report.warnings)


def test_format_report_matches_expected_structure(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=2, ny=2, material=material, thickness=0.01)
    report = generate_validation_report(mesh)
    text = format_report(report)

    assert "Mesh Validation Report" in text
    assert "Status: OK" in text
    assert "Nodes:" in text
    assert "Elements:" in text
    assert "Quality:" in text


def test_format_report_includes_warnings_section() -> None:
    material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=2, ny=2, material=material, thickness=0.01)
    mesh.add_node(Node(id=9999, x=100.0, y=100.0, z=0.0))
    report = generate_validation_report(mesh)
    text = format_report(report)
    assert "Warnings:" in text
