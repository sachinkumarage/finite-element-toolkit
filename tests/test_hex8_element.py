"""Tests for the Hex8Element3D domain object."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis.dof import TranslationDOF
from femtoolkit.analysis.element import AssemblableElement
from femtoolkit.exceptions import DegenerateElementError, ValidationError
from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Node


@pytest.fixture
def material() -> LinearElastic3D:
    return LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)


@pytest.fixture
def unit_cube_nodes() -> tuple[Node, ...]:
    coords = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ]
    return tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(coords))


def test_hex8_satisfies_assemblable_protocol(
    unit_cube_nodes: tuple[Node, ...], material: LinearElastic3D
) -> None:
    element = Hex8Element3D(id=1, nodes=unit_cube_nodes, material=material)
    assert isinstance(element, AssemblableElement)
    assert element.dofs_per_node == 3


def test_volume_of_unit_cube(unit_cube_nodes: tuple[Node, ...], material: LinearElastic3D) -> None:
    element = Hex8Element3D(id=1, nodes=unit_cube_nodes, material=material)
    assert_allclose(element.volume, 1.0)


def test_volume_of_scaled_cuboid(material: LinearElastic3D) -> None:
    coords = [
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (2.0, 3.0, 0.0),
        (0.0, 3.0, 0.0),
        (0.0, 0.0, 4.0),
        (2.0, 0.0, 4.0),
        (2.0, 3.0, 4.0),
        (0.0, 3.0, 4.0),
    ]
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(coords))
    element = Hex8Element3D(id=1, nodes=nodes, material=material)
    assert_allclose(element.volume, 2.0 * 3.0 * 4.0, rtol=1e-10)


def test_dof_keys_order(unit_cube_nodes: tuple[Node, ...], material: LinearElastic3D) -> None:
    element = Hex8Element3D(id=1, nodes=unit_cube_nodes, material=material)
    keys = element.dof_keys()
    assert len(keys) == 24
    assert keys[0] == (1, TranslationDOF.X)
    assert keys[1] == (1, TranslationDOF.Y)
    assert keys[2] == (1, TranslationDOF.Z)
    assert keys[-1] == (8, TranslationDOF.Z)


def test_shape_functions_sum_to_one_at_gauss_points() -> None:
    from femtoolkit.continuum.gauss import GAUSS_2X2X2_POINTS
    from femtoolkit.continuum.shape_functions import hex8_shape_functions

    for point in GAUSS_2X2X2_POINTS:
        n_values = hex8_shape_functions(point.xi, point.eta, point.zeta)
        assert sum(n_values) == pytest.approx(1.0)


def test_stiffness_matrix_symmetric_and_correct_shape(
    unit_cube_nodes: tuple[Node, ...], material: LinearElastic3D
) -> None:
    element = Hex8Element3D(id=1, nodes=unit_cube_nodes, material=material)
    stiffness = element.stiffness_matrix
    assert stiffness.shape == (24, 24)
    assert_allclose(stiffness, stiffness.T)


def test_stiffness_matrix_has_exactly_six_rigid_body_modes(
    unit_cube_nodes: tuple[Node, ...], material: LinearElastic3D
) -> None:
    element = Hex8Element3D(id=1, nodes=unit_cube_nodes, material=material)
    eigenvalues = np.linalg.eigvalsh(element.stiffness_matrix)

    near_zero = np.sum(np.abs(eigenvalues) < 1e-6 * np.max(np.abs(eigenvalues)))
    assert near_zero == 6
    assert np.all(eigenvalues > -1e-3 * np.max(np.abs(eigenvalues)))


def test_uniform_strain_response(
    unit_cube_nodes: tuple[Node, ...], material: LinearElastic3D
) -> None:
    """A linear nodal displacement field must reproduce the exact constant strain,
    even though HEX8's B matrix itself varies pointwise -- the *interpolated*
    strain of a purely linear field is still exactly constant everywhere."""
    element = Hex8Element3D(id=1, nodes=unit_cube_nodes, material=material)

    a = np.array([0.004, -0.002, 0.0015])
    b = np.array([0.001, 0.003, -0.001])
    c = np.array([-0.0005, 0.0007, 0.002])

    displacements = []
    for node in unit_cube_nodes:
        displacements.extend(
            [
                a[0] * node.x + a[1] * node.y + a[2] * node.z,
                b[0] * node.x + b[1] * node.y + b[2] * node.z,
                c[0] * node.x + c[1] * node.y + c[2] * node.z,
            ]
        )

    strain = element.strain_from_dofs(displacements)
    expected = np.array([a[0], b[1], c[2], a[1] + b[0], b[2] + c[1], a[2] + c[0]])
    assert_allclose(strain, expected, atol=1e-10)


def test_inverted_node_order_raises_degenerate_element_error(material: LinearElastic3D) -> None:
    coords = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ]
    nodes = list(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(coords))
    # Swap two nodes on the bottom face to invert the winding.
    nodes[1], nodes[3] = nodes[3], nodes[1]
    with pytest.raises(DegenerateElementError):
        Hex8Element3D(id=1, nodes=tuple(nodes), material=material)


def test_rejects_wrong_material_type(unit_cube_nodes: tuple[Node, ...]) -> None:
    from femtoolkit.materials import LinearElastic2D

    bad_material = LinearElastic2D(
        youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress"
    )
    with pytest.raises(ValidationError):
        Hex8Element3D(id=1, nodes=unit_cube_nodes, material=bad_material)


def test_rejects_wrong_node_count(material: LinearElastic3D) -> None:
    nodes = tuple(Node(id=i + 1, x=float(i), y=0.0, z=0.0) for i in range(7))
    with pytest.raises(ValidationError):
        Hex8Element3D(id=1, nodes=nodes, material=material)
