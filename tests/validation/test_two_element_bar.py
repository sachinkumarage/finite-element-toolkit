"""Validation Case 2: two-element bar chain.

Model: Node 1 ---- Element 1 ---- Node 2 ---- Element 2 ---- Node 3.
Both elements: E = 200 GPa, A = 0.01 m^2, L = 1 m each.
Boundary condition: u1 = 0.
Load: F = 1000 N at node 3 (none at node 2).

Since the only external load is applied at the free end and no load is
applied at the shared node, statics requires the internal axial force to
be the same in both elements and equal to the applied end load:

    N1 = N2 = F = 1000 N

From N, each element's stress, strain, and displacement follow
independently: sigma = N / A, epsilon = sigma / E, and node
displacements accumulate as u_(i+1) = u_i + epsilon_i * L_i.
"""

from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis
from femtoolkit.materials import Material
from femtoolkit.mesh import BarElement, Mesh, Node
from femtoolkit.sections import CrossSection

YOUNGS_MODULUS = 200e9
AREA = 0.01
LENGTH_1 = 1.0
LENGTH_2 = 1.0
APPLIED_FORCE = 1000.0


def _build_result():
    steel = Material(
        name="Steel", density=7850.0, youngs_modulus=YOUNGS_MODULUS, poissons_ratio=0.3
    )
    section = CrossSection(area=AREA)

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=LENGTH_1, y=0.0, z=0.0)
    node_3 = Node(id=3, x=LENGTH_1 + LENGTH_2, y=0.0, z=0.0)

    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    mesh.add_node(node_3)
    mesh.add_element(
        BarElement(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)
    )
    mesh.add_element(
        BarElement(id=2, nodes=(node_2, node_3), material=steel, cross_section=section)
    )

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=0, value=0.0))
    analysis.add_load(NodalLoad(node_id=3, dof=0, value=APPLIED_FORCE))
    return analysis.solve()


def test_two_element_bar_axial_force_is_uniform() -> None:
    """No intermediate load: both elements carry the same axial force as the applied load."""
    result = _build_result()

    assert_allclose(result.element_axial_force(1), APPLIED_FORCE, rtol=1e-10)
    assert_allclose(result.element_axial_force(2), APPLIED_FORCE, rtol=1e-10)


def test_two_element_bar_stress_and_strain() -> None:
    result = _build_result()

    expected_stress = APPLIED_FORCE / AREA
    expected_strain = expected_stress / YOUNGS_MODULUS

    assert_allclose(result.element_stress(1), expected_stress, rtol=1e-10)
    assert_allclose(result.element_stress(2), expected_stress, rtol=1e-10)
    assert_allclose(result.element_strain(1), expected_strain, rtol=1e-10)
    assert_allclose(result.element_strain(2), expected_strain, rtol=1e-10)


def test_two_element_bar_displacements() -> None:
    result = _build_result()

    expected_strain = (APPLIED_FORCE / AREA) / YOUNGS_MODULUS
    expected_u2 = expected_strain * LENGTH_1
    expected_u3 = expected_u2 + expected_strain * LENGTH_2

    assert_allclose(result.displacement(1), 0.0, atol=1e-12)
    assert_allclose(result.displacement(2), expected_u2, rtol=1e-10)
    assert_allclose(result.displacement(3), expected_u3, rtol=1e-10)


def test_two_element_bar_reaction_satisfies_equilibrium() -> None:
    result = _build_result()

    assert_allclose(result.reaction(1), -APPLIED_FORCE, rtol=1e-10)
    assert_allclose(APPLIED_FORCE + result.reaction(1), 0.0, atol=1e-6)
