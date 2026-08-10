"""Validation Case 1: horizontal 2D truss (backward validation against the 1D bar).

Geometry: Node 1 --------- Node 2, L = 2 m, lying along the global X axis.
Properties: E = 200 GPa, A = 0.01 m^2.
Boundary conditions: Node 1 pinned (ux=uy=0), Node 2 rollered (uy=0).
Load: Fx = 1000 N at node 2.

A horizontal 2D truss member under a purely axial load must reproduce
the Version 3 1D bar solution exactly (direction cosines c=1, s=0 reduce
the 2D stiffness matrix to the 1D bar matrix -- see
tests/test_stiffness.py::test_truss_stiffness_2d_horizontal_reduces_to_bar).
This is an important backward-compatibility check for the shared
numerical infrastructure introduced in Version 4.

Analytical displacement: ux2 = F * L / (E * A) (Version 3's formula).
"""

from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import Material
from femtoolkit.mesh import Mesh, Node, TrussElement2D
from femtoolkit.sections import CrossSection

YOUNGS_MODULUS = 200e9
AREA = 0.01
LENGTH = 2.0
APPLIED_FORCE = 1000.0


def _build_result():
    steel = Material(
        name="Steel", density=7850.0, youngs_modulus=YOUNGS_MODULUS, poissons_ratio=0.3
    )
    section = CrossSection(area=AREA)
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=LENGTH, y=0.0, z=0.0)
    element = TrussElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)

    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    mesh.add_element(element)

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.X, value=0.0))
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.Y, value=0.0))
    analysis.add_boundary_condition(BoundaryCondition(node_id=2, dof=TranslationDOF.Y, value=0.0))
    analysis.add_load(NodalLoad(node_id=2, dof=TranslationDOF.X, value=APPLIED_FORCE))
    return analysis.solve()


def test_horizontal_truss_matches_1d_bar_displacement() -> None:
    """The 2D result must reproduce the Version 3 analytical bar formula u = FL/(EA)."""
    result = _build_result()

    analytical_displacement = APPLIED_FORCE * LENGTH / (YOUNGS_MODULUS * AREA)
    ux2, uy2 = result.node_displacement(2)

    assert_allclose(ux2, analytical_displacement, rtol=1e-10)
    assert_allclose(uy2, 0.0, atol=1e-12)


def test_horizontal_truss_reactions_and_equilibrium() -> None:
    result = _build_result()

    rx1, ry1 = result.node_reaction(1)
    rx2, ry2 = result.node_reaction(2)

    assert_allclose(rx1, -APPLIED_FORCE, rtol=1e-10)
    assert_allclose(ry1, 0.0, atol=1e-6)
    assert_allclose(ry2, 0.0, atol=1e-6)

    # Global equilibrium: sum of applied forces + reactions ~ 0 in both directions.
    assert_allclose(APPLIED_FORCE + rx1 + rx2, 0.0, atol=1e-6)
    assert_allclose(ry1 + ry2, 0.0, atol=1e-6)


def test_horizontal_truss_strain_stress_and_axial_force() -> None:
    result = _build_result()

    expected_stress = APPLIED_FORCE / AREA
    expected_strain = expected_stress / YOUNGS_MODULUS

    assert_allclose(result.element_strain(1), expected_strain, rtol=1e-10)
    assert_allclose(result.element_stress(1), expected_stress, rtol=1e-10)
    assert_allclose(result.element_axial_force(1), APPLIED_FORCE, rtol=1e-10)
