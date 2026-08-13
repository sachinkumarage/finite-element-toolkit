"""Tests for the QuadElement2D domain object."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis.dof import TranslationDOF
from femtoolkit.analysis.element import AssemblableElement, ContinuumElement, StructuralElement
from femtoolkit.exceptions import DegenerateElementError, ValidationError
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Node, QuadElement2D


@pytest.fixture
def material() -> LinearElastic2D:
    return LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")


@pytest.fixture
def square_nodes() -> tuple[Node, Node, Node, Node]:
    return (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=1.0, y=0.0, z=0.0),
        Node(id=3, x=1.0, y=1.0, z=0.0),
        Node(id=4, x=0.0, y=1.0, z=0.0),
    )


def test_quad_element_satisfies_assemblable_and_continuum_protocols(
    square_nodes: tuple[Node, Node, Node, Node], material: LinearElastic2D
) -> None:
    element = QuadElement2D(id=1, nodes=square_nodes, material=material, thickness=0.01)

    assert isinstance(element, AssemblableElement)
    assert isinstance(element, ContinuumElement)
    assert not isinstance(element, StructuralElement)
    assert element.dofs_per_node == 2


def test_quad_element_node_and_material_association(
    square_nodes: tuple[Node, Node, Node, Node], material: LinearElastic2D
) -> None:
    element = QuadElement2D(id=1, nodes=square_nodes, material=material, thickness=0.01)

    assert element.nodes == square_nodes
    assert element.material is material
    assert element.thickness == 0.01


def test_area_for_unit_square(
    square_nodes: tuple[Node, Node, Node, Node], material: LinearElastic2D
) -> None:
    element = QuadElement2D(id=1, nodes=square_nodes, material=material, thickness=0.01)

    assert_allclose(element.area, 1.0)


def test_area_for_rectangle(material: LinearElastic2D) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=3.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=3.0, y=2.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=2.0, z=0.0)
    element = QuadElement2D(
        id=1, nodes=(node_1, node_2, node_3, node_4), material=material, thickness=0.01
    )

    assert_allclose(element.area, 6.0)


def test_dof_keys_order(material: LinearElastic2D) -> None:
    node_1 = Node(id=5, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=9, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=7, x=1.0, y=1.0, z=0.0)
    node_4 = Node(id=3, x=0.0, y=1.0, z=0.0)
    element = QuadElement2D(
        id=1, nodes=(node_1, node_2, node_3, node_4), material=material, thickness=0.01
    )

    assert element.dof_keys() == (
        (5, TranslationDOF.X),
        (5, TranslationDOF.Y),
        (9, TranslationDOF.X),
        (9, TranslationDOF.Y),
        (7, TranslationDOF.X),
        (7, TranslationDOF.Y),
        (3, TranslationDOF.X),
        (3, TranslationDOF.Y),
    )


def test_centroid_b_matrix_shape(
    square_nodes: tuple[Node, Node, Node, Node], material: LinearElastic2D
) -> None:
    element = QuadElement2D(id=1, nodes=square_nodes, material=material, thickness=0.01)

    assert element.centroid_b_matrix.shape == (3, 8)


def test_stiffness_matrix_shape_and_symmetry(
    square_nodes: tuple[Node, Node, Node, Node], material: LinearElastic2D
) -> None:
    element = QuadElement2D(id=1, nodes=square_nodes, material=material, thickness=0.01)

    stiffness = element.stiffness_matrix
    assert stiffness.shape == (8, 8)
    assert_allclose(stiffness, stiffness.T)


def test_stiffness_matrix_matches_direct_formula(
    square_nodes: tuple[Node, Node, Node, Node], material: LinearElastic2D
) -> None:
    element = QuadElement2D(id=1, nodes=square_nodes, material=material, thickness=0.02)

    from femtoolkit.analysis.stiffness import quad_element_stiffness

    expected = quad_element_stiffness(
        element._x_coords, element._y_coords, 0.02, material.constitutive_matrix
    )
    assert_allclose(element.stiffness_matrix, expected)


def test_strain_from_dofs_constant_field(
    square_nodes: tuple[Node, Node, Node, Node], material: LinearElastic2D
) -> None:
    """u=ax+by, v=cx+dy must give strain [a, d, b+c] exactly, since a
    linear field is a special case of the Q4's bilinear interpolation.
    """
    element = QuadElement2D(id=1, nodes=square_nodes, material=material, thickness=0.01)

    a, b, c, d = 0.01, 0.02, -0.005, 0.015
    displacements = []
    for node in square_nodes:
        displacements.append(a * node.x + b * node.y)
        displacements.append(c * node.x + d * node.y)

    strain = element.strain_from_dofs(displacements)
    assert_allclose(strain, [a, d, b + c], atol=1e-10)


def test_stress_from_dofs_matches_constitutive_matrix(
    square_nodes: tuple[Node, Node, Node, Node], material: LinearElastic2D
) -> None:
    element = QuadElement2D(id=1, nodes=square_nodes, material=material, thickness=0.01)
    displacements = [0.0, 0.0, 0.001, 0.0, 0.001, 0.0005, 0.0, 0.0005]

    strain = element.strain_from_dofs(displacements)
    stress = element.stress_from_dofs(displacements)

    assert_allclose(stress, material.constitutive_matrix @ strain)


def test_von_mises_plane_stress_dispatch(
    square_nodes: tuple[Node, Node, Node, Node], material: LinearElastic2D
) -> None:
    element = QuadElement2D(id=1, nodes=square_nodes, material=material, thickness=0.01)
    displacements = [0.0, 0.0, 0.001, 0.0, 0.001, 0.0005, 0.0, 0.0005]

    from femtoolkit.continuum.stress import von_mises_plane_stress

    sigma_x, sigma_y, tau_xy = element.stress_from_dofs(displacements)
    expected = von_mises_plane_stress(sigma_x, sigma_y, tau_xy)
    assert_allclose(element.von_mises_from_dofs(displacements), expected)


def test_von_mises_plane_strain_dispatch(square_nodes: tuple[Node, Node, Node, Node]) -> None:
    plane_strain_material = LinearElastic2D(
        youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_strain"
    )
    element = QuadElement2D(
        id=1, nodes=square_nodes, material=plane_strain_material, thickness=0.01
    )
    displacements = [0.0, 0.0, 0.001, 0.0, 0.001, 0.0005, 0.0, 0.0005]

    from femtoolkit.continuum.stress import von_mises_plane_strain

    sigma_x, sigma_y, tau_xy = element.stress_from_dofs(displacements)
    expected = von_mises_plane_strain(sigma_x, sigma_y, tau_xy, plane_strain_material.poisson_ratio)
    assert_allclose(element.von_mises_from_dofs(displacements), expected)


def test_principal_stresses_from_dofs(
    square_nodes: tuple[Node, Node, Node, Node], material: LinearElastic2D
) -> None:
    element = QuadElement2D(id=1, nodes=square_nodes, material=material, thickness=0.01)
    displacements = [0.0, 0.0, 0.001, 0.0, 0.001, 0.0005, 0.0, 0.0005]

    from femtoolkit.continuum.stress import principal_stresses_2d

    sigma_x, sigma_y, tau_xy = element.stress_from_dofs(displacements)
    expected = principal_stresses_2d(sigma_x, sigma_y, tau_xy)
    assert_allclose(element.principal_stresses_from_dofs(displacements), expected)


def test_invalid_id_raises(
    square_nodes: tuple[Node, Node, Node, Node], material: LinearElastic2D
) -> None:
    with pytest.raises(ValidationError):
        QuadElement2D(id=0, nodes=square_nodes, material=material, thickness=0.01)


def test_three_nodes_raises(material: LinearElastic2D) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=1.0, y=1.0, z=0.0)

    with pytest.raises(ValidationError):
        QuadElement2D(id=1, nodes=(node_1, node_2, node_3), material=material, thickness=0.01)  # type: ignore[arg-type]


def test_duplicate_node_reference_raises(material: LinearElastic2D) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=1.0, y=1.0, z=0.0)

    with pytest.raises(ValidationError):
        QuadElement2D(
            id=1, nodes=(node_1, node_1, node_2, node_3), material=material, thickness=0.01
        )


def test_missing_material_raises(square_nodes: tuple[Node, Node, Node, Node]) -> None:
    with pytest.raises(ValidationError):
        QuadElement2D(id=1, nodes=square_nodes, material=None, thickness=0.01)  # type: ignore[arg-type]


def test_bar_material_rejected(square_nodes: tuple[Node, Node, Node, Node]) -> None:
    """A plain Material (Version 1) is not a LinearElastic2D and must be rejected."""
    from femtoolkit.materials import Material

    bar_material = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)

    with pytest.raises(ValidationError):
        QuadElement2D(id=1, nodes=square_nodes, material=bar_material, thickness=0.01)  # type: ignore[arg-type]


@pytest.mark.parametrize("thickness", [0.0, -0.01, float("nan"), float("inf")])
def test_invalid_thickness_raises(
    square_nodes: tuple[Node, Node, Node, Node], material: LinearElastic2D, thickness: float
) -> None:
    with pytest.raises(ValidationError):
        QuadElement2D(id=1, nodes=square_nodes, material=material, thickness=thickness)


def test_collinear_nodes_raise_degenerate_error(material: LinearElastic2D) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=2.0, y=0.0, z=0.0)
    node_4 = Node(id=4, x=3.0, y=0.0, z=0.0)

    with pytest.raises(DegenerateElementError):
        QuadElement2D(
            id=1, nodes=(node_1, node_2, node_3, node_4), material=material, thickness=0.01
        )


def test_clockwise_node_order_raises_degenerate_error(material: LinearElastic2D) -> None:
    """Q4 node order is fixed (counter-clockwise); reversing it gives a
    negative Jacobian determinant everywhere, which is rejected.
    """
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=0.0, y=1.0, z=0.0)
    node_3 = Node(id=3, x=1.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=1.0, y=0.0, z=0.0)

    with pytest.raises(DegenerateElementError):
        QuadElement2D(
            id=1, nodes=(node_1, node_2, node_3, node_4), material=material, thickness=0.01
        )


def test_self_intersecting_quad_raises_degenerate_error(material: LinearElastic2D) -> None:
    """Nodes 2 and 4 swapped relative to a valid square create a
    self-intersecting (bowtie) quadrilateral.
    """
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=0.0, y=1.0, z=0.0)
    node_3 = Node(id=3, x=1.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=1.0, y=0.0, z=0.0)

    with pytest.raises(DegenerateElementError):
        QuadElement2D(
            id=1, nodes=(node_1, node_3, node_2, node_4), material=material, thickness=0.01
        )


def test_non_right_angle_quad_is_accepted(material: LinearElastic2D) -> None:
    """A general (non-rectangular) convex quadrilateral must be accepted."""
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=2.5, y=1.5, z=0.0)
    node_4 = Node(id=4, x=0.3, y=1.2, z=0.0)
    element = QuadElement2D(
        id=1, nodes=(node_1, node_2, node_3, node_4), material=material, thickness=0.01
    )

    assert element.area > 0.0
    assert np.all(np.isfinite(element.stiffness_matrix))
