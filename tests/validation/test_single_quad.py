"""Validation: single Q4 element under uniaxial tension, plus an
independently derived stiffness matrix.

Geometry:

    Node4 -------- Node3
     |              |
     |              |
     |              |
    Node1 -------- Node2

    Node 1 = (0, 0), Node 2 = (1, 0), Node 3 = (1, 1), Node 4 = (0, 1)
    E = 200 GPa, v = 0.3 (plane stress), t = 0.01 m
    Node 1: fixed (ux = uy = 0), Node 4: rollered (ux = 0)
    Fx = 2500 N each at Node 2 and Node 3 (total 5000 N, equivalent to a
    uniform 500 kPa tensile traction on the right edge, since
    Fx_total / (H*t) = 5000 N / (1 m * 0.01 m) = 500000 Pa)

This validates the complete Q4 workflow -- geometry, shape functions,
Jacobian, Gauss-integrated stiffness, assembly, solving, and strain/stress
recovery -- against the known uniaxial stress state, and independently
verifies the stiffness matrix itself (Case 4 of the CST validation suite,
repeated for Q4) against a from-scratch NumPy re-derivation of the
isoparametric formula that does not call any of
:mod:`femtoolkit.continuum` or :mod:`femtoolkit.analysis.stiffness`.
"""

import numpy as np
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, Node, QuadElement2D

YOUNGS_MODULUS = 200e9
POISSON_RATIO = 0.3
THICKNESS = 0.01
APPLIED_LOAD = 5000.0  # N, total, split evenly between Node 2 and Node 3

EXPECTED_SIGMA_X = APPLIED_LOAD / (1.0 * THICKNESS)  # height = 1 m


def _build_result():
    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, formulation="plane_stress"
    )
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=1.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=1.0, z=0.0)
    element = QuadElement2D(
        id=1, nodes=(node_1, node_2, node_3, node_4), material=material, thickness=THICKNESS
    )

    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4):
        mesh.add_node(node)
    mesh.add_element(element)

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.X, value=0.0))
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.Y, value=0.0))
    analysis.add_boundary_condition(BoundaryCondition(node_id=4, dof=TranslationDOF.X, value=0.0))
    analysis.add_load(NodalLoad(node_id=2, dof=TranslationDOF.X, value=APPLIED_LOAD / 2))
    analysis.add_load(NodalLoad(node_id=3, dof=TranslationDOF.X, value=APPLIED_LOAD / 2))
    return analysis.solve()


def test_displacement_matches_uniaxial_theory() -> None:
    result = _build_result()

    epsilon_x = EXPECTED_SIGMA_X / YOUNGS_MODULUS
    epsilon_y = -POISSON_RATIO * epsilon_x

    ux2, uy2 = result.node_displacement(2)
    ux3, uy3 = result.node_displacement(3)
    assert_allclose(ux2, epsilon_x * 1.0, rtol=1e-6)
    assert_allclose(ux3, epsilon_x * 1.0, rtol=1e-6)
    assert_allclose(uy3, epsilon_y * 1.0, rtol=1e-6)
    assert_allclose(uy2, 0.0, atol=1e-12)


def test_strain_matches_uniaxial_theory() -> None:
    result = _build_result()

    epsilon_x, epsilon_y, gamma_xy = result.element_strain(1)
    assert_allclose(epsilon_x, EXPECTED_SIGMA_X / YOUNGS_MODULUS, rtol=1e-6)
    assert_allclose(epsilon_y, -POISSON_RATIO * EXPECTED_SIGMA_X / YOUNGS_MODULUS, rtol=1e-6)
    assert_allclose(gamma_xy, 0.0, atol=1e-9)


def test_stress_matches_uniaxial_theory() -> None:
    result = _build_result()

    sigma_x, sigma_y, tau_xy = result.element_stress(1)
    assert_allclose(sigma_x, EXPECTED_SIGMA_X, rtol=1e-6)
    assert_allclose(sigma_y, 0.0, atol=1e-3)
    assert_allclose(tau_xy, 0.0, atol=1e-3)


def test_reactions_balance_the_applied_load() -> None:
    result = _build_result()

    rx1, ry1 = result.node_reaction(1)
    rx4, ry4 = result.node_reaction(4)
    assert_allclose(rx1 + rx4 + APPLIED_LOAD, 0.0, atol=1e-6)
    assert_allclose(ry1 + ry4, 0.0, atol=1e-6)


def _independently_derived_stiffness() -> np.ndarray:
    """Re-derive the unit-square Q4 stiffness matrix (E=1, v=0.3, t=1)
    from scratch, without calling femtoolkit.continuum or
    femtoolkit.analysis.stiffness.
    """
    e, v, t = 1.0, 0.3, 1.0
    x = np.array([0.0, 1.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    factor = e / (1.0 - v**2)
    d_matrix = factor * np.array([[1.0, v, 0.0], [v, 1.0, 0.0], [0.0, 0.0, (1.0 - v) / 2.0]])

    abscissa = 1.0 / np.sqrt(3.0)
    points = [
        (-abscissa, -abscissa),
        (abscissa, -abscissa),
        (abscissa, abscissa),
        (-abscissa, abscissa),
    ]

    stiffness = np.zeros((8, 8))
    for xi, eta in points:
        dn_dxi = np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]) / 4.0
        dn_deta = np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]) / 4.0
        jacobian = np.array([[dn_dxi @ x, dn_dxi @ y], [dn_deta @ x, dn_deta @ y]])
        det_j = jacobian[0, 0] * jacobian[1, 1] - jacobian[0, 1] * jacobian[1, 0]
        j_inv = np.linalg.inv(jacobian)
        physical = j_inv @ np.array([dn_dxi, dn_deta])
        dn_dx, dn_dy = physical[0], physical[1]

        b_matrix = np.zeros((3, 8))
        for i in range(4):
            b_matrix[0, 2 * i] = dn_dx[i]
            b_matrix[1, 2 * i + 1] = dn_dy[i]
            b_matrix[2, 2 * i] = dn_dy[i]
            b_matrix[2, 2 * i + 1] = dn_dx[i]

        stiffness += 1.0 * t * (b_matrix.T @ d_matrix @ b_matrix) * det_j

    return stiffness


def test_stiffness_matrix_matches_independent_derivation() -> None:
    material = LinearElastic2D(youngs_modulus=1.0, poisson_ratio=0.3, formulation="plane_stress")
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=1.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=1.0, z=0.0)
    element = QuadElement2D(
        id=1, nodes=(node_1, node_2, node_3, node_4), material=material, thickness=1.0
    )

    assert_allclose(element.stiffness_matrix, _independently_derived_stiffness(), atol=1e-9)
