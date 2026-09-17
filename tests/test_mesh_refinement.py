"""Tests for femtoolkit.mesh.refinement.refine_uniform (Version 25)."""

import pytest

from femtoolkit.exceptions import UnsupportedRefinementError
from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D, Material
from femtoolkit.mesh import BarElement, Mesh, Node
from femtoolkit.mesh.generator import create_quad_mesh, create_triangular_mesh
from femtoolkit.mesh.refinement import refine_uniform
from femtoolkit.mesh.validation import generate_validation_report, validate_mesh
from femtoolkit.sections import CrossSection


@pytest.fixture
def material() -> LinearElastic2D:
    return LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")


def test_uniform_quad_refinement_quadruples_elements(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01)
    refined = refine_uniform(mesh)

    assert len(refined.elements) == 4 * len(mesh.elements)
    validate_mesh(refined)


def test_uniform_cst_refinement_quadruples_elements(material: LinearElastic2D) -> None:
    mesh = create_triangular_mesh(
        width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01
    )
    refined = refine_uniform(mesh)

    assert len(refined.elements) == 4 * len(mesh.elements)
    validate_mesh(refined)


def test_refinement_preserves_total_area_quad(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01)
    refined = refine_uniform(mesh)

    original_area = sum(element.area for element in mesh.elements)
    refined_area = sum(element.area for element in refined.elements)
    assert refined_area == pytest.approx(original_area)


def test_refinement_preserves_total_area_cst(material: LinearElastic2D) -> None:
    mesh = create_triangular_mesh(
        width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01
    )
    refined = refine_uniform(mesh)

    original_area = sum(element.area for element in mesh.elements)
    refined_area = sum(element.area for element in refined.elements)
    assert refined_area == pytest.approx(original_area)


def test_refinement_never_mutates_the_original_mesh(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01)
    original_node_count = len(mesh.nodes)
    original_element_count = len(mesh.elements)

    refine_uniform(mesh)

    assert len(mesh.nodes) == original_node_count
    assert len(mesh.elements) == original_element_count


def test_refinement_preserves_boundary_nodes(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01)
    refined = refine_uniform(mesh)

    rectangle = Rectangle(width=2.0, height=1.0)
    left_boundary = rectangle.boundary("left")
    left_before = {round(node.y, 9) for node in mesh.nodes_on_boundary(left_boundary)}
    left_after = {round(node.y, 9) for node in refined.nodes_on_boundary(left_boundary)}

    assert left_before.issubset(left_after)
    assert len(left_after) == 2 * len(left_before) - 1


def test_refinement_preserves_material_and_thickness(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=2, ny=2, material=material, thickness=0.03)
    refined = refine_uniform(mesh)

    for element in refined.elements:
        assert element.material is material
        assert element.thickness == pytest.approx(0.03)


def test_shared_edges_do_not_create_duplicate_midpoints(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=2, ny=1, material=material, thickness=0.01)
    refined = refine_uniform(mesh)

    coordinates = [(round(n.x, 9), round(n.y, 9), round(n.z, 9)) for n in refined.nodes]
    assert len(coordinates) == len(set(coordinates))


def test_local_refinement_refines_only_named_elements(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01)
    refined = refine_uniform(mesh, elements=[1])

    assert len(refined.elements) == len(mesh.elements) - 1 + 4
    # A hanging-node mesh is still geometrically well-formed (no duplicate
    # coordinates, no invalid connectivity) even though it is non-conforming.
    report = generate_validation_report(refined)
    assert report.num_duplicate_nodes == 0
    assert report.num_invalid_connectivity == 0


def test_double_refinement(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=2, ny=1, material=material, thickness=0.01)
    twice = refine_uniform(refine_uniform(mesh))

    assert len(twice.elements) == 16 * len(mesh.elements)
    validate_mesh(twice)


def test_refine_unsupported_element_type_raises() -> None:
    mesh = Mesh()
    n1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    n2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    mesh.add_node(n1)
    mesh.add_node(n2)
    bar_material = Material(name="steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    bar = BarElement(
        id=1, nodes=(n1, n2), material=bar_material, cross_section=CrossSection(area=0.01)
    )
    mesh.add_element(bar)

    with pytest.raises(UnsupportedRefinementError):
        refine_uniform(mesh, elements=[1])


def test_refine_nonexistent_element_id_raises(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=2, ny=2, material=material, thickness=0.01)
    from femtoolkit.exceptions import EntityNotFoundError

    with pytest.raises(EntityNotFoundError):
        refine_uniform(mesh, elements=[9999])
