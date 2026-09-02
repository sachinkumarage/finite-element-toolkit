"""Tests for the Tet4Element3D domain object."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis.dof import TranslationDOF
from femtoolkit.analysis.element import AssemblableElement
from femtoolkit.exceptions import DegenerateElementError, ValidationError
from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Node, Tet4Element3D


@pytest.fixture
def material() -> LinearElastic3D:
    return LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)


@pytest.fixture
def unit_tet_nodes() -> tuple[Node, Node, Node, Node]:
    return (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=1.0, y=0.0, z=0.0),
        Node(id=3, x=0.0, y=1.0, z=0.0),
        Node(id=4, x=0.0, y=0.0, z=1.0),
    )


def test_tet4_satisfies_assemblable_protocol(
    unit_tet_nodes: tuple[Node, Node, Node, Node], material: LinearElastic3D
) -> None:
    element = Tet4Element3D(id=1, nodes=unit_tet_nodes, material=material)

    assert isinstance(element, AssemblableElement)
    assert element.dofs_per_node == 3


def test_volume_of_unit_right_tetrahedron(
    unit_tet_nodes: tuple[Node, Node, Node, Node], material: LinearElastic3D
) -> None:
    element = Tet4Element3D(id=1, nodes=unit_tet_nodes, material=material)
    assert_allclose(element.volume, 1.0 / 6.0)
    assert_allclose(element.signed_volume, 1.0 / 6.0)


def test_reordered_nodes_give_same_absolute_volume(
    unit_tet_nodes: tuple[Node, Node, Node, Node], material: LinearElastic3D
) -> None:
    n1, n2, n3, n4 = unit_tet_nodes
    reordered = Tet4Element3D(id=1, nodes=(n1, n3, n2, n4), material=material)

    assert_allclose(reordered.volume, 1.0 / 6.0)
    # The sign flips under an odd permutation of the four nodes.
    assert reordered.signed_volume < 0.0


def test_dof_keys_order(material: LinearElastic3D) -> None:
    n1 = Node(id=5, x=0.0, y=0.0, z=0.0)
    n2 = Node(id=9, x=1.0, y=0.0, z=0.0)
    n3 = Node(id=7, x=0.0, y=1.0, z=0.0)
    n4 = Node(id=2, x=0.0, y=0.0, z=1.0)
    element = Tet4Element3D(id=1, nodes=(n1, n2, n3, n4), material=material)

    assert element.dof_keys() == (
        (5, TranslationDOF.X),
        (5, TranslationDOF.Y),
        (5, TranslationDOF.Z),
        (9, TranslationDOF.X),
        (9, TranslationDOF.Y),
        (9, TranslationDOF.Z),
        (7, TranslationDOF.X),
        (7, TranslationDOF.Y),
        (7, TranslationDOF.Z),
        (2, TranslationDOF.X),
        (2, TranslationDOF.Y),
        (2, TranslationDOF.Z),
    )


def test_b_matrix_shape_and_partition_of_unity_gradient_sum(
    unit_tet_nodes: tuple[Node, Node, Node, Node], material: LinearElastic3D
) -> None:
    """The four shape function gradients must sum to zero (partition of unity)."""
    element = Tet4Element3D(id=1, nodes=unit_tet_nodes, material=material)
    b_matrix = element.b_matrix
    assert b_matrix.shape == (6, 12)

    dn_dx = b_matrix[0, 0::3]
    dn_dy = b_matrix[1, 1::3]
    dn_dz = b_matrix[2, 2::3]
    assert_allclose(dn_dx.sum(), 0.0, atol=1e-10)
    assert_allclose(dn_dy.sum(), 0.0, atol=1e-10)
    assert_allclose(dn_dz.sum(), 0.0, atol=1e-10)


def test_stiffness_matrix_symmetric_and_correct_shape(
    unit_tet_nodes: tuple[Node, Node, Node, Node], material: LinearElastic3D
) -> None:
    element = Tet4Element3D(id=1, nodes=unit_tet_nodes, material=material)
    stiffness = element.stiffness_matrix

    assert stiffness.shape == (12, 12)
    assert_allclose(stiffness, stiffness.T)


def test_stiffness_matrix_has_exactly_six_rigid_body_modes(
    unit_tet_nodes: tuple[Node, Node, Node, Node], material: LinearElastic3D
) -> None:
    element = Tet4Element3D(id=1, nodes=unit_tet_nodes, material=material)
    eigenvalues = np.linalg.eigvalsh(element.stiffness_matrix)

    near_zero = np.sum(np.abs(eigenvalues) < 1e-6 * np.max(np.abs(eigenvalues)))
    assert near_zero == 6
    assert np.all(eigenvalues > -1e-3 * np.max(np.abs(eigenvalues)))  # no negative modes


def test_uniform_strain_response(
    unit_tet_nodes: tuple[Node, Node, Node, Node], material: LinearElastic3D
) -> None:
    """A linear nodal displacement field must reproduce the exact constant strain.

    See tests/validation/test_tet4_patch.py for the full-analysis-level version.
    """
    element = Tet4Element3D(id=1, nodes=unit_tet_nodes, material=material)

    a = np.array([0.004, -0.002, 0.0015])
    b = np.array([0.001, 0.003, -0.001])
    c = np.array([-0.0005, 0.0007, 0.002])

    displacements = []
    for node in unit_tet_nodes:
        displacements.extend(
            [
                a[0] * node.x + a[1] * node.y + a[2] * node.z,
                b[0] * node.x + b[1] * node.y + b[2] * node.z,
                c[0] * node.x + c[1] * node.y + c[2] * node.z,
            ]
        )

    strain = element.strain_from_dofs(displacements)
    expected = np.array([a[0], b[1], c[2], a[1] + b[0], b[2] + c[1], a[2] + c[0]])
    assert_allclose(strain, expected, atol=1e-13)


def test_degenerate_coplanar_element_raises(material: LinearElastic3D) -> None:
    n1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    n2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    n3 = Node(id=3, x=2.0, y=0.0, z=0.0)
    n4 = Node(id=4, x=3.0, y=0.0, z=0.0)  # all four nodes coplanar (collinear even)
    with pytest.raises(DegenerateElementError):
        Tet4Element3D(id=1, nodes=(n1, n2, n3, n4), material=material)


def test_rejects_wrong_material_type(unit_tet_nodes: tuple[Node, Node, Node, Node]) -> None:
    from femtoolkit.materials import LinearElastic2D

    bad_material = LinearElastic2D(
        youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress"
    )
    with pytest.raises(ValidationError):
        Tet4Element3D(id=1, nodes=unit_tet_nodes, material=bad_material)


def test_rejects_duplicate_nodes(material: LinearElastic3D) -> None:
    n1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    n2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    n3 = Node(id=3, x=0.0, y=1.0, z=0.0)
    with pytest.raises(ValidationError):
        Tet4Element3D(id=1, nodes=(n1, n2, n3, n1), material=material)
