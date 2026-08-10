"""Validation Case 4: two truss elements with different materials and areas.

Model: Node 1 ---- Element 1 (Steel) ---- Node 2 ---- Element 2 (Aluminum) ---- Node 3,
all collinear along the global X axis.
Boundary conditions: Node 1 pinned (ux=uy=0), Node 2 and Node 3 rollered (uy=0).
Load: Fx = 5000 N at Node 3 (none at Node 2).

As in the Version 3 multi-material bar validation, statics requires the
internal axial force to be the same in both elements (N1 = N2 = F) since
no load is applied at the shared node -- even though each element's
stress and strain differ, because they use their own E and A.
"""

from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import Material
from femtoolkit.mesh import Mesh, Node, TrussElement2D
from femtoolkit.sections import CrossSection

STEEL_E = 200e9
STEEL_A = 0.01
ALUMINUM_E = 70e9
ALUMINUM_A = 0.02
LENGTH_1 = 1.0
LENGTH_2 = 1.0
APPLIED_FORCE = 5000.0


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

    element_1 = TrussElement2D(
        id=1, nodes=(node_1, node_2), material=steel, cross_section=steel_section
    )
    element_2 = TrussElement2D(
        id=2, nodes=(node_2, node_3), material=aluminum, cross_section=aluminum_section
    )

    mesh = Mesh()
    for node in (node_1, node_2, node_3):
        mesh.add_node(node)
    mesh.add_element(element_1)
    mesh.add_element(element_2)

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.X, value=0.0))
    for node_id in (1, 2, 3):
        analysis.add_boundary_condition(
            BoundaryCondition(node_id=node_id, dof=TranslationDOF.Y, value=0.0)
        )
    analysis.add_load(NodalLoad(node_id=3, dof=TranslationDOF.X, value=APPLIED_FORCE))
    return analysis.solve()


def test_multi_material_truss_axial_force_is_uniform() -> None:
    result = _build_result()

    assert_allclose(result.element_axial_force(1), APPLIED_FORCE, rtol=1e-9)
    assert_allclose(result.element_axial_force(2), APPLIED_FORCE, rtol=1e-9)


def test_multi_material_truss_stress_differs_by_area() -> None:
    result = _build_result()

    assert_allclose(result.element_stress(1), APPLIED_FORCE / STEEL_A, rtol=1e-9)
    assert_allclose(result.element_stress(2), APPLIED_FORCE / ALUMINUM_A, rtol=1e-9)
    assert result.element_stress(1) != result.element_stress(2)


def test_multi_material_truss_strain_differs_by_modulus() -> None:
    result = _build_result()

    expected_strain_steel = (APPLIED_FORCE / STEEL_A) / STEEL_E
    expected_strain_aluminum = (APPLIED_FORCE / ALUMINUM_A) / ALUMINUM_E

    assert_allclose(result.element_strain(1), expected_strain_steel, rtol=1e-9)
    assert_allclose(result.element_strain(2), expected_strain_aluminum, rtol=1e-9)


def test_multi_material_truss_reaction_and_equilibrium() -> None:
    result = _build_result()

    rx1, _ = result.node_reaction(1)
    assert_allclose(rx1, -APPLIED_FORCE, rtol=1e-9)
    assert_allclose(APPLIED_FORCE + rx1, 0.0, atol=1e-6)
