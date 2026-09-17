"""Tests for femtoolkit.mesh.quality.evaluator.QualityEvaluator (Version 25)."""

import pytest

from femtoolkit.exceptions import UnsupportedQualityMetricError, ValidationError
from femtoolkit.materials import LinearElastic2D, LinearElastic3D, Material
from femtoolkit.mesh import BarElement, Hex8Element3D, Mesh, Node
from femtoolkit.mesh.generator import create_quad_mesh, create_triangular_mesh
from femtoolkit.mesh.quality.evaluator import QualityEvaluator
from femtoolkit.sections import CrossSection


@pytest.fixture
def material_2d() -> LinearElastic2D:
    return LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")


def test_evaluate_regular_quad_mesh(material_2d: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=4, ny=2, material=material_2d, thickness=0.01)
    report = QualityEvaluator().evaluate(mesh)

    assert report.num_elements_evaluated == 8
    assert report.minimum_quality == pytest.approx(1.0)
    assert report.maximum_quality == pytest.approx(1.0)
    assert report.mean_quality == pytest.approx(1.0)
    assert report.poor_quality_element_ids == []
    assert not report.has_poor_quality_elements


def test_evaluate_regular_triangular_mesh(material_2d: LinearElastic2D) -> None:
    mesh = create_triangular_mesh(
        width=2.0, height=1.0, nx=4, ny=2, material=material_2d, thickness=0.01
    )
    report = QualityEvaluator().evaluate(mesh)
    assert report.num_elements_evaluated == 16


def test_evaluate_3d_mesh() -> None:
    material = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    coords = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(coords))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=material)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    report = QualityEvaluator().evaluate(mesh)
    assert report.num_elements_evaluated == 1
    assert report.minimum_quality == pytest.approx(1.0)


def test_poor_quality_elements_identified(material_2d: LinearElastic2D) -> None:
    nodes = (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=10.0, y=0.0, z=0.0),
        Node(id=3, x=10.0, y=0.1, z=0.0),
        Node(id=4, x=0.0, y=0.1, z=0.0),
    )
    from femtoolkit.mesh.quad_element import QuadElement2D

    elongated = QuadElement2D(id=1, nodes=nodes, material=material_2d, thickness=0.01)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(elongated)

    report = QualityEvaluator(poor_quality_threshold=0.5).evaluate(mesh)
    assert report.has_poor_quality_elements
    assert report.poor_quality_element_ids == [1]
    assert report.warnings


def test_evaluate_skips_elements_with_no_quality_concept(material_2d: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=2, ny=2, material=material_2d, thickness=0.01)
    bar_material = Material(
        name="steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3
    )
    n1 = Node(id=100, x=5.0, y=5.0, z=0.0)
    n2 = Node(id=101, x=6.0, y=5.0, z=0.0)
    mesh.add_node(n1)
    mesh.add_node(n2)
    bar = BarElement(
        id=100, nodes=(n1, n2), material=bar_material, cross_section=CrossSection(area=0.01)
    )
    mesh.add_element(bar)

    report = QualityEvaluator().evaluate(mesh)
    assert report.num_elements_evaluated == 4  # the bar is excluded, not an error


def test_evaluate_raises_when_no_evaluable_elements() -> None:
    mesh = Mesh()
    n1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    n2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    mesh.add_node(n1)
    mesh.add_node(n2)
    bar_material = Material(
        name="steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3
    )
    bar = BarElement(
        id=1, nodes=(n1, n2), material=bar_material, cross_section=CrossSection(area=0.01)
    )
    mesh.add_element(bar)

    with pytest.raises(UnsupportedQualityMetricError):
        QualityEvaluator().evaluate(mesh)


def test_invalid_poor_quality_threshold_raises() -> None:
    with pytest.raises(ValidationError):
        QualityEvaluator(poor_quality_threshold=0.0)
    with pytest.raises(ValidationError):
        QualityEvaluator(poor_quality_threshold=1.5)
    with pytest.raises(ValidationError):
        QualityEvaluator(poor_quality_threshold=-0.1)


def test_default_poor_quality_threshold_is_reasonable() -> None:
    evaluator = QualityEvaluator()
    assert 0.0 < evaluator.poor_quality_threshold <= 1.0
