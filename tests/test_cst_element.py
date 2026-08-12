"""Tests for the CSTElement2D domain object."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis.dof import TranslationDOF
from femtoolkit.analysis.element import AssemblableElement, ContinuumElement
from femtoolkit.exceptions import DegenerateElementError, ValidationError
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import CSTElement2D, Node


@pytest.fixture
def material() -> LinearElastic2D:
    return LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")


@pytest.fixture
def triangle_nodes() -> tuple[Node, Node, Node]:
    return (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=1.0, y=0.0, z=0.0),
        Node(id=3, x=0.0, y=1.0, z=0.0),
    )


def test_cst_element_satisfies_assemblable_and_continuum_protocols(
    triangle_nodes: tuple[Node, Node, Node], material: LinearElastic2D
) -> None:
    element = CSTElement2D(id=1, nodes=triangle_nodes, material=material, thickness=0.01)

    assert isinstance(element, AssemblableElement)
    assert isinstance(element, ContinuumElement)
    assert element.dofs_per_node == 2


def test_cst_element_node_and_material_association(
    triangle_nodes: tuple[Node, Node, Node], material: LinearElastic2D
) -> None:
    element = CSTElement2D(id=1, nodes=triangle_nodes, material=material, thickness=0.01)

    assert element.nodes == triangle_nodes
    assert element.material is material
    assert element.thickness == 0.01


def test_area(triangle_nodes: tuple[Node, Node, Node], material: LinearElastic2D) -> None:
    element = CSTElement2D(id=1, nodes=triangle_nodes, material=material, thickness=0.01)

    assert_allclose(element.area, 0.5)
    assert_allclose(element.signed_area, 0.5)


def test_clockwise_nodes_give_positive_area(material: LinearElastic2D) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=0.0, y=1.0, z=0.0)
    node_3 = Node(id=3, x=1.0, y=0.0, z=0.0)
    element = CSTElement2D(id=1, nodes=(node_1, node_2, node_3), material=material, thickness=0.01)

    assert element.signed_area < 0.0
    assert_allclose(element.area, 0.5)  # always positive, regardless of winding


def test_dof_keys_order(material: LinearElastic2D) -> None:
    node_1 = Node(id=5, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=9, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=7, x=0.0, y=1.0, z=0.0)
    element = CSTElement2D(id=1, nodes=(node_1, node_2, node_3), material=material, thickness=0.01)

    assert element.dof_keys() == (
        (5, TranslationDOF.X),
        (5, TranslationDOF.Y),
        (9, TranslationDOF.X),
        (9, TranslationDOF.Y),
        (7, TranslationDOF.X),
        (7, TranslationDOF.Y),
    )


def test_b_matrix_shape(triangle_nodes: tuple[Node, Node, Node], material: LinearElastic2D) -> None:
    element = CSTElement2D(id=1, nodes=triangle_nodes, material=material, thickness=0.01)

    assert element.b_matrix.shape == (3, 6)


def test_stiffness_matrix_shape_and_symmetry(
    triangle_nodes: tuple[Node, Node, Node], material: LinearElastic2D
) -> None:
    element = CSTElement2D(id=1, nodes=triangle_nodes, material=material, thickness=0.01)

    stiffness = element.stiffness_matrix
    assert stiffness.shape == (6, 6)
    assert_allclose(stiffness, stiffness.T)


def test_stiffness_matrix_matches_direct_formula(
    triangle_nodes: tuple[Node, Node, Node], material: LinearElastic2D
) -> None:
    element = CSTElement2D(id=1, nodes=triangle_nodes, material=material, thickness=0.02)

    from femtoolkit.analysis.stiffness import cst_element_stiffness

    expected = cst_element_stiffness(
        thickness=0.02,
        area=element.area,
        b_matrix=element.b_matrix,
        d_matrix=material.constitutive_matrix,
    )
    assert_allclose(element.stiffness_matrix, expected)


def test_stiffness_matrix_independent_of_node_winding_order(material: LinearElastic2D) -> None:
    """The same physical triangle, listed clockwise or counter-clockwise,
    must have the same total stiffness (sum of all entries is winding-
    independent since it reflects the same physical rigidity)."""
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=0.0, y=1.0, z=0.0)
    ccw = CSTElement2D(id=1, nodes=(node_1, node_2, node_3), material=material, thickness=0.01)
    cw = CSTElement2D(id=2, nodes=(node_1, node_3, node_2), material=material, thickness=0.01)

    # Both stiffness matrices must be positive semidefinite with the same eigenvalues.
    eig_ccw = np.sort(np.linalg.eigvalsh(ccw.stiffness_matrix))
    eig_cw = np.sort(np.linalg.eigvalsh(cw.stiffness_matrix))
    assert_allclose(eig_ccw, eig_cw, atol=1e-3)


def test_strain_from_dofs_constant_field(
    triangle_nodes: tuple[Node, Node, Node], material: LinearElastic2D
) -> None:
    """u=ax+by, v=cx+dy must give strain [a, d, b+c] exactly (see tests/test_strain.py)."""
    element = CSTElement2D(id=1, nodes=triangle_nodes, material=material, thickness=0.01)

    a, b, c, d = 0.01, 0.02, -0.005, 0.015
    displacements = []
    for node in triangle_nodes:
        displacements.append(a * node.x + b * node.y)
        displacements.append(c * node.x + d * node.y)

    strain = element.strain_from_dofs(displacements)
    assert_allclose(strain, [a, d, b + c], atol=1e-12)


def test_stress_from_dofs_matches_constitutive_matrix(
    triangle_nodes: tuple[Node, Node, Node], material: LinearElastic2D
) -> None:
    element = CSTElement2D(id=1, nodes=triangle_nodes, material=material, thickness=0.01)
    displacements = [0.0, 0.0, 0.001, 0.0, 0.0, 0.0005]

    strain = element.strain_from_dofs(displacements)
    stress = element.stress_from_dofs(displacements)

    assert_allclose(stress, material.constitutive_matrix @ strain)


def test_von_mises_plane_stress_dispatch(
    triangle_nodes: tuple[Node, Node, Node], material: LinearElastic2D
) -> None:
    element = CSTElement2D(id=1, nodes=triangle_nodes, material=material, thickness=0.01)
    displacements = [0.0, 0.0, 0.001, 0.0, 0.0, 0.0005]

    from femtoolkit.continuum.stress import von_mises_plane_stress

    sigma_x, sigma_y, tau_xy = element.stress_from_dofs(displacements)
    expected = von_mises_plane_stress(sigma_x, sigma_y, tau_xy)
    assert_allclose(element.von_mises_from_dofs(displacements), expected)


def test_von_mises_plane_strain_dispatch(triangle_nodes: tuple[Node, Node, Node]) -> None:
    plane_strain_material = LinearElastic2D(
        youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_strain"
    )
    element = CSTElement2D(
        id=1, nodes=triangle_nodes, material=plane_strain_material, thickness=0.01
    )
    displacements = [0.0, 0.0, 0.001, 0.0, 0.0, 0.0005]

    from femtoolkit.continuum.stress import von_mises_plane_strain

    sigma_x, sigma_y, tau_xy = element.stress_from_dofs(displacements)
    expected = von_mises_plane_strain(sigma_x, sigma_y, tau_xy, plane_strain_material.poisson_ratio)
    assert_allclose(element.von_mises_from_dofs(displacements), expected)


def test_principal_stresses_from_dofs(
    triangle_nodes: tuple[Node, Node, Node], material: LinearElastic2D
) -> None:
    element = CSTElement2D(id=1, nodes=triangle_nodes, material=material, thickness=0.01)
    displacements = [0.0, 0.0, 0.001, 0.0, 0.0, 0.0005]

    from femtoolkit.continuum.stress import principal_stresses_2d

    sigma_x, sigma_y, tau_xy = element.stress_from_dofs(displacements)
    expected = principal_stresses_2d(sigma_x, sigma_y, tau_xy)
    assert_allclose(element.principal_stresses_from_dofs(displacements), expected)


def test_invalid_id_raises(
    triangle_nodes: tuple[Node, Node, Node], material: LinearElastic2D
) -> None:
    with pytest.raises(ValidationError):
        CSTElement2D(id=0, nodes=triangle_nodes, material=material, thickness=0.01)


def test_two_nodes_raises(material: LinearElastic2D) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)

    with pytest.raises(ValidationError):
        CSTElement2D(id=1, nodes=(node_1, node_2), material=material, thickness=0.01)  # type: ignore[arg-type]


def test_duplicate_node_reference_raises(material: LinearElastic2D) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)

    with pytest.raises(ValidationError):
        CSTElement2D(id=1, nodes=(node_1, node_1, node_2), material=material, thickness=0.01)


def test_missing_material_raises(triangle_nodes: tuple[Node, Node, Node]) -> None:
    with pytest.raises(ValidationError):
        CSTElement2D(id=1, nodes=triangle_nodes, material=None, thickness=0.01)  # type: ignore[arg-type]


def test_bar_material_rejected(triangle_nodes: tuple[Node, Node, Node]) -> None:
    """A plain Material (Version 1) is not a LinearElastic2D and must be rejected."""
    from femtoolkit.materials import Material

    bar_material = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)

    with pytest.raises(ValidationError):
        CSTElement2D(id=1, nodes=triangle_nodes, material=bar_material, thickness=0.01)  # type: ignore[arg-type]


@pytest.mark.parametrize("thickness", [0.0, -0.01, float("nan"), float("inf")])
def test_invalid_thickness_raises(
    triangle_nodes: tuple[Node, Node, Node], material: LinearElastic2D, thickness: float
) -> None:
    with pytest.raises(ValidationError):
        CSTElement2D(id=1, nodes=triangle_nodes, material=material, thickness=thickness)


def test_collinear_nodes_raise_degenerate_error(material: LinearElastic2D) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=1.0, z=0.0)
    node_3 = Node(id=3, x=2.0, y=2.0, z=0.0)

    with pytest.raises(DegenerateElementError):
        CSTElement2D(id=1, nodes=(node_1, node_2, node_3), material=material, thickness=0.01)


def test_nearly_collinear_nodes_raise_degenerate_error(material: LinearElastic2D) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=0.5, y=1e-14, z=0.0)

    with pytest.raises(DegenerateElementError):
        CSTElement2D(id=1, nodes=(node_1, node_2, node_3), material=material, thickness=0.01)


def test_duplicate_coordinates_raise_degenerate_error(material: LinearElastic2D) -> None:
    node_1 = Node(id=1, x=1.0, y=1.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=1.0, z=0.0)
    node_3 = Node(id=3, x=2.0, y=2.0, z=0.0)

    with pytest.raises(DegenerateElementError):
        CSTElement2D(id=1, nodes=(node_1, node_2, node_3), material=material, thickness=0.01)
