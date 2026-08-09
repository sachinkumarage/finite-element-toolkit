"""Validation Case 1: single fixed-free axial bar.

Geometry: Node 1 --------- Node 2, L = 2 m.
Properties: E = 200 GPa, A = 0.01 m^2.
Boundary condition: u1 = 0 (fixed).
Load: F = 1000 N at node 2.

Analytical displacement: u2 = F * L / (E * A).

This also checks global equilibrium (applied load + reaction ~ 0) and the
linear-elastic energy identity U = W for a fixed (zero-displacement)
support, where:

    U = 1/2 * {u}^T [K] {u}   (internal strain energy)
    W = 1/2 * {u}^T {F}       (external work)
"""

from numpy.testing import assert_allclose

from femtoolkit.analysis import (
    BoundaryCondition,
    DOFMap,
    ElementStiffnessContribution,
    NodalLoad,
    StaticLinearAnalysis,
    assemble_global_stiffness,
    build_force_vector,
)
from femtoolkit.materials import Material
from femtoolkit.mesh import BarElement, Mesh, Node
from femtoolkit.sections import CrossSection

YOUNGS_MODULUS = 200e9
AREA = 0.01
LENGTH = 2.0
APPLIED_FORCE = 1000.0


def _build_analysis() -> tuple[StaticLinearAnalysis, Node, Node]:
    steel = Material(
        name="Steel", density=7850.0, youngs_modulus=YOUNGS_MODULUS, poissons_ratio=0.3
    )
    section = CrossSection(area=AREA)
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=LENGTH, y=0.0, z=0.0)

    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    mesh.add_element(
        BarElement(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)
    )

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(node_id=node_1.id, dof=0, value=0.0))
    analysis.add_load(NodalLoad(node_id=node_2.id, dof=0, value=APPLIED_FORCE))
    return analysis, node_1, node_2


def test_single_bar_matches_analytical_displacement() -> None:
    analysis, node_1, node_2 = _build_analysis()
    result = analysis.solve()

    analytical_displacement = APPLIED_FORCE * LENGTH / (YOUNGS_MODULUS * AREA)
    assert_allclose(result.displacement(node_1.id), 0.0, atol=1e-12)
    assert_allclose(result.displacement(node_2.id), analytical_displacement, rtol=1e-10)


def test_single_bar_reaction_satisfies_equilibrium() -> None:
    analysis, node_1, node_2 = _build_analysis()
    result = analysis.solve()

    assert_allclose(result.reaction(node_1.id), -APPLIED_FORCE, rtol=1e-10)
    # Global equilibrium: applied load + reaction ~ 0.
    assert_allclose(APPLIED_FORCE + result.reaction(node_1.id), 0.0, atol=1e-6)


def test_single_bar_strain_stress_and_axial_force() -> None:
    analysis, node_1, node_2 = _build_analysis()
    result = analysis.solve()

    expected_stress = APPLIED_FORCE / AREA
    expected_strain = expected_stress / YOUNGS_MODULUS

    assert_allclose(result.element_strain(1), expected_strain, rtol=1e-10)
    assert_allclose(result.element_stress(1), expected_stress, rtol=1e-10)
    assert_allclose(result.element_axial_force(1), APPLIED_FORCE, rtol=1e-10)


def test_single_bar_energy_identity() -> None:
    """U = 1/2 u^T K u equals W = 1/2 u^T F for a zero-displacement support.

    This holds because, at equilibrium, [K]{u} = {F} + {R} (see
    AnalysisResult.reaction). Since the prescribed displacement is zero at
    every constrained DOF, u^T R = 0, so u^T K u reduces to u^T F.
    """
    steel = Material(
        name="Steel", density=7850.0, youngs_modulus=YOUNGS_MODULUS, poissons_ratio=0.3
    )
    section = CrossSection(area=AREA)
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=LENGTH, y=0.0, z=0.0)
    bar = BarElement(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)

    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    stiffness = assemble_global_stiffness(
        dof_map, [ElementStiffnessContribution((1, 2), bar.stiffness_matrix)]
    )
    applied_forces = build_force_vector(dof_map, [NodalLoad(node_id=2, dof=0, value=APPLIED_FORCE)])

    analysis, _, _ = _build_analysis()
    result = analysis.solve()
    u = result.displacements

    strain_energy = 0.5 * u @ stiffness @ u
    external_work = 0.5 * u @ applied_forces

    assert_allclose(strain_energy, external_work, rtol=1e-10)
