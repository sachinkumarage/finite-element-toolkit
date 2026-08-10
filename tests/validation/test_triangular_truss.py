"""Validation Case 2: triangular truss, independently solved by method of joints.

Geometry (a "tent" triangle, 45-degree diagonals):

          Node 3 (1, 1)
           /\\
          /  \\
         /    \\
        /      \\
Node 1 ---------- Node 2
 (0,0)   L=2m     (2,0)

Members: base (Node1-Node2), left diagonal (Node1-Node3), right diagonal
(Node2-Node3). Supports: Node1 pinned (ux=uy=0), Node2 rollered (uy=0).
Load: Fy = -1000 N at Node3 (downward).

This structure is statically determinate (m=3, r=3, n=3: m+r=2n). The
member forces and reactions below were derived independently by the
method of joints (not by running the solver) and are used as the
expected values:

Joint 3 (apex), with N_left = N_right = N by symmetry of the two 45-degree
diagonals:

    -N/sqrt(2) + N/sqrt(2) = 0                (horizontal equilibrium; trivially satisfied)
    -(N + N)/sqrt(2) - P = 0  =>  N = -P/sqrt(2)   (vertical equilibrium)

Joint 1 (pin), vertical equilibrium: Ry1 + N_left/sqrt(2) = 0 => Ry1 = P/2.
Joint 2 (roller), vertical equilibrium: Ry2 + N_right/sqrt(2) = 0 => Ry2 = P/2.
Joint 2, horizontal equilibrium: -N_base - N_right/sqrt(2) = 0 => N_base = P/2.
Global horizontal equilibrium (no applied Fx) gives Rx1 = 0.

So: N_base = P/2 (tension), N_left = N_right = -P/sqrt(2) (compression),
Rx1 = 0, Ry1 = Ry2 = P/2.

The vertical displacement at Node 3 is cross-checked with the unit-load
(virtual work) method, using these same hand-derived member forces:

    delta_y3 = -(N_base^2 * L_base + N_left^2 * L_left + N_right^2 * L_right) / (P * E * A)
"""

import math

from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import Material
from femtoolkit.mesh import Mesh, Node, TrussElement2D
from femtoolkit.sections import CrossSection

YOUNGS_MODULUS = 200e9
AREA = 0.001
APPLIED_LOAD = 1000.0

BASE_LENGTH = 2.0
DIAGONAL_LENGTH = math.sqrt(2)

# Independently derived (method of joints), not computed by the solver.
EXPECTED_N_BASE = APPLIED_LOAD / 2
EXPECTED_N_LEFT = -APPLIED_LOAD / math.sqrt(2)
EXPECTED_N_RIGHT = -APPLIED_LOAD / math.sqrt(2)
EXPECTED_RX1 = 0.0
EXPECTED_RY1 = APPLIED_LOAD / 2
EXPECTED_RY2 = APPLIED_LOAD / 2


def _build_result():
    steel = Material(
        name="Steel", density=7850.0, youngs_modulus=YOUNGS_MODULUS, poissons_ratio=0.3
    )
    section = CrossSection(area=AREA)

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=1.0, y=1.0, z=0.0)

    base = TrussElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)
    left = TrussElement2D(id=2, nodes=(node_1, node_3), material=steel, cross_section=section)
    right = TrussElement2D(id=3, nodes=(node_2, node_3), material=steel, cross_section=section)

    mesh = Mesh()
    for node in (node_1, node_2, node_3):
        mesh.add_node(node)
    for element in (base, left, right):
        mesh.add_element(element)

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.X, value=0.0))
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.Y, value=0.0))
    analysis.add_boundary_condition(BoundaryCondition(node_id=2, dof=TranslationDOF.Y, value=0.0))
    analysis.add_load(NodalLoad(node_id=3, dof=TranslationDOF.Y, value=-APPLIED_LOAD))
    return analysis.solve()


def test_triangular_truss_member_forces() -> None:
    result = _build_result()

    assert_allclose(result.element_axial_force(1), EXPECTED_N_BASE, rtol=1e-9)
    assert_allclose(result.element_axial_force(2), EXPECTED_N_LEFT, rtol=1e-9)
    assert_allclose(result.element_axial_force(3), EXPECTED_N_RIGHT, rtol=1e-9)


def test_triangular_truss_tension_and_compression_signs() -> None:
    result = _build_result()

    assert result.element_axial_force(1) > 0  # base: tension
    assert result.element_axial_force(2) < 0  # left diagonal: compression
    assert result.element_axial_force(3) < 0  # right diagonal: compression


def test_triangular_truss_reactions() -> None:
    result = _build_result()

    rx1, ry1 = result.node_reaction(1)
    rx2, ry2 = result.node_reaction(2)

    assert_allclose(rx1, EXPECTED_RX1, atol=1e-6)
    assert_allclose(ry1, EXPECTED_RY1, rtol=1e-9)
    assert_allclose(ry2, EXPECTED_RY2, rtol=1e-9)


def test_triangular_truss_global_equilibrium() -> None:
    result = _build_result()

    rx1, ry1 = result.node_reaction(1)
    rx2, ry2 = result.node_reaction(2)

    assert_allclose(rx1 + rx2, 0.0, atol=1e-6)  # sum Fx = 0 (no applied horizontal load)
    assert_allclose(ry1 + ry2 - APPLIED_LOAD, 0.0, atol=1e-6)  # sum Fy = 0


def test_triangular_truss_apex_displacement_matches_virtual_work() -> None:
    """Cross-check via the unit-load (virtual work) method, using the
    independently hand-derived member forces above -- not the solver's
    own results -- as a second, analytically independent computation
    path.
    """
    result = _build_result()

    expected_uy3 = -(
        EXPECTED_N_BASE**2 * BASE_LENGTH
        + EXPECTED_N_LEFT**2 * DIAGONAL_LENGTH
        + EXPECTED_N_RIGHT**2 * DIAGONAL_LENGTH
    ) / (APPLIED_LOAD * YOUNGS_MODULUS * AREA)

    _, uy3 = result.node_displacement(3)
    assert_allclose(uy3, expected_uy3, rtol=1e-9)
    assert uy3 < 0  # apex moves downward under the downward load
