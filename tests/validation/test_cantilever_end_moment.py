"""Validation Case 3: cantilever beam with a pure end moment.

    Fixed                    Free
    |-------------------------o
    Node 1        L = 2 m     Node 2
                               Mz = 500 N*m

    E = 200 GPa, I = 8.333e-6 m^4

Analytical solution (Euler-Bernoulli beam theory):

    theta = M * L / (E * I)         (tip rotation)
    delta = M * L^2 / (2 * E * I)   (tip deflection)

and the fixed-end reaction moment exactly balances the applied moment,
Mz_reaction = -M, since no other load acts on the beam.
"""

from numpy.testing import assert_allclose

from femtoolkit.analysis import (
    BoundaryCondition,
    NodalLoad,
    RotationDOF,
    StaticLinearAnalysis,
    TranslationDOF,
)
from femtoolkit.materials import Material
from femtoolkit.mesh import FrameElement2D, Mesh, Node
from femtoolkit.sections import CrossSection

YOUNGS_MODULUS = 200e9
AREA = 0.01
SECOND_MOMENT_OF_AREA = 8.333e-6
LENGTH = 2.0
APPLIED_MOMENT = 500.0

ANALYTICAL_TIP_ROTATION = APPLIED_MOMENT * LENGTH / (YOUNGS_MODULUS * SECOND_MOMENT_OF_AREA)
ANALYTICAL_TIP_DEFLECTION = (
    APPLIED_MOMENT * LENGTH**2 / (2 * YOUNGS_MODULUS * SECOND_MOMENT_OF_AREA)
)


def _build_result():
    steel = Material(
        name="Steel", density=7850.0, youngs_modulus=YOUNGS_MODULUS, poissons_ratio=0.3
    )
    section = CrossSection(area=AREA, second_moment_of_area=SECOND_MOMENT_OF_AREA)

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=LENGTH, y=0.0, z=0.0)
    element = FrameElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)

    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    mesh.add_element(element)

    analysis = StaticLinearAnalysis(mesh)
    for dof in (TranslationDOF.X, TranslationDOF.Y, RotationDOF.RZ):
        analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=dof, value=0.0))
    analysis.add_load(NodalLoad(node_id=2, dof=RotationDOF.RZ, value=APPLIED_MOMENT))
    return analysis.solve()


def test_tip_rotation_matches_analytical_solution() -> None:
    result = _build_result()

    assert_allclose(result.displacement(2, RotationDOF.RZ), ANALYTICAL_TIP_ROTATION, rtol=1e-9)


def test_tip_deflection_matches_analytical_solution() -> None:
    result = _build_result()

    assert_allclose(result.displacement(2, TranslationDOF.Y), ANALYTICAL_TIP_DEFLECTION, rtol=1e-9)


def test_reaction_moment_balances_applied_moment() -> None:
    result = _build_result()

    rx, ry, mz = result.node_reaction(1)
    assert_allclose(rx, 0.0, atol=1e-6)
    assert_allclose(ry, 0.0, atol=1e-6)
    assert_allclose(mz, -APPLIED_MOMENT, rtol=1e-9)


def test_no_shear_is_induced_by_a_pure_end_moment() -> None:
    """A pure end moment produces no shear force anywhere in the beam."""
    result = _build_result()

    forces = result.element_end_forces(1)
    assert_allclose(forces.node_1.shear_force, 0.0, atol=1e-6)
    assert_allclose(forces.node_2.shear_force, 0.0, atol=1e-6)
