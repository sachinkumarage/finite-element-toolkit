"""Validation Case 4: two elements with different materials and areas.

Model: Node 1 ---- Element 1 (Steel) ---- Node 2 ---- Element 2 (Aluminum) ---- Node 3.
Boundary condition: u1 = 0.
Load: F = 1000 N at node 3 (none at node 2).

As in the two-element validation case, statics requires the internal
axial force to be the same in both elements (N1 = N2 = F), since no load
is applied at the shared node. Each element's stress and strain still
differ because they use their own E and A:

    sigma_i = N / A_i
    epsilon_i = sigma_i / E_i
"""

from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis
from femtoolkit.materials import Material
from femtoolkit.mesh import BarElement, Mesh, Node
from femtoolkit.sections import CrossSection

STEEL_E = 200e9
STEEL_A = 0.01
ALUMINUM_E = 70e9
ALUMINUM_A = 0.02
LENGTH_1 = 1.0
LENGTH_2 = 1.0
APPLIED_FORCE = 1000.0


def _build_result():
    steel = Material(name="Steel", density=7850.0, youngs_modulus=STEEL_E, poissons_ratio=0.3)
    aluminum = Material(
        name="Aluminum", density=2700.0, youngs_modulus=ALUMINUM_E, poissons_ratio=0.33
    )
    steel_section = CrossSection(area=STEEL_A)
    aluminum_section = CrossSection(area=ALUMINUM_A)

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=LENGTH_1, y=0.0, z=0.0)
    node_3 = Node(id=3, x=LENGTH_1 + LENGTH_2, y=0.0, z=0.0)

    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    mesh.add_node(node_3)
    mesh.add_element(
        BarElement(id=1, nodes=(node_1, node_2), material=steel, cross_section=steel_section)
    )
    mesh.add_element(
        BarElement(id=2, nodes=(node_2, node_3), material=aluminum, cross_section=aluminum_section)
    )

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=0, value=0.0))
    analysis.add_load(NodalLoad(node_id=3, dof=0, value=APPLIED_FORCE))
    return analysis.solve()


def test_multi_material_axial_force_is_uniform() -> None:
    result = _build_result()

    assert_allclose(result.element_axial_force(1), APPLIED_FORCE, rtol=1e-10)
    assert_allclose(result.element_axial_force(2), APPLIED_FORCE, rtol=1e-10)


def test_multi_material_stress_differs_by_area() -> None:
    result = _build_result()

    assert_allclose(result.element_stress(1), APPLIED_FORCE / STEEL_A, rtol=1e-10)
    assert_allclose(result.element_stress(2), APPLIED_FORCE / ALUMINUM_A, rtol=1e-10)
    assert result.element_stress(1) != result.element_stress(2)


def test_multi_material_strain_differs_by_modulus() -> None:
    result = _build_result()

    expected_strain_steel = (APPLIED_FORCE / STEEL_A) / STEEL_E
    expected_strain_aluminum = (APPLIED_FORCE / ALUMINUM_A) / ALUMINUM_E

    assert_allclose(result.element_strain(1), expected_strain_steel, rtol=1e-10)
    assert_allclose(result.element_strain(2), expected_strain_aluminum, rtol=1e-10)
    assert result.element_strain(1) != result.element_strain(2)


def test_multi_material_displacements_accumulate_along_the_chain() -> None:
    result = _build_result()

    expected_strain_steel = (APPLIED_FORCE / STEEL_A) / STEEL_E
    expected_strain_aluminum = (APPLIED_FORCE / ALUMINUM_A) / ALUMINUM_E
    expected_u2 = expected_strain_steel * LENGTH_1
    expected_u3 = expected_u2 + expected_strain_aluminum * LENGTH_2

    assert_allclose(result.displacement(2), expected_u2, rtol=1e-10)
    assert_allclose(result.displacement(3), expected_u3, rtol=1e-10)
