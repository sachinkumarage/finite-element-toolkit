"""Validation Case 2: cantilever beam with a transverse tip load.

The classical cantilever, modeled with a single frame element:

    Fixed                         Free
    |-----------------------------o
    Node 1          L = 2 m       Node 2
                                   Fy = -1000 N

    E = 200 GPa, I = 8.333e-6 m^4

Analytical solution (Euler-Bernoulli beam theory):

    delta = P * L^3 / (3 * E * I)   (tip deflection, downward)
    theta = P * L^2 / (2 * E * I)   (tip rotation)
    M     = P * L                  (fixed-end moment magnitude)
    V     = P                      (fixed-end shear magnitude)
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
TIP_LOAD = 1000.0

ANALYTICAL_TIP_DEFLECTION = -TIP_LOAD * LENGTH**3 / (3 * YOUNGS_MODULUS * SECOND_MOMENT_OF_AREA)
ANALYTICAL_TIP_ROTATION = -TIP_LOAD * LENGTH**2 / (2 * YOUNGS_MODULUS * SECOND_MOMENT_OF_AREA)
ANALYTICAL_FIXED_END_MOMENT = TIP_LOAD * LENGTH
ANALYTICAL_FIXED_END_SHEAR = TIP_LOAD


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
    analysis.add_load(NodalLoad(node_id=2, dof=TranslationDOF.Y, value=-TIP_LOAD))
    return analysis.solve()


def test_tip_deflection_matches_analytical_solution() -> None:
    result = _build_result()

    assert_allclose(
        result.displacement(2, TranslationDOF.Y), ANALYTICAL_TIP_DEFLECTION, rtol=1e-9
    )


def test_tip_rotation_matches_analytical_solution() -> None:
    result = _build_result()

    assert_allclose(result.displacement(2, RotationDOF.RZ), ANALYTICAL_TIP_ROTATION, rtol=1e-9)


def test_fixed_end_moment_matches_analytical_solution() -> None:
    result = _build_result()

    assert_allclose(result.element_bending_moment(1), ANALYTICAL_FIXED_END_MOMENT, rtol=1e-9)


def test_fixed_end_shear_matches_analytical_solution() -> None:
    result = _build_result()

    assert_allclose(result.element_shear_force(1), ANALYTICAL_FIXED_END_SHEAR, rtol=1e-9)


def test_free_end_has_zero_moment() -> None:
    """No load or support acts at the free tip, so its bending moment must be zero."""
    result = _build_result()

    assert_allclose(result.element_bending_moment(1, end="node_2"), 0.0, atol=1e-6)


def test_reactions_satisfy_global_equilibrium() -> None:
    result = _build_result()

    rx, ry, mz = result.node_reaction(1)
    assert_allclose(rx, 0.0, atol=1e-6)
    assert_allclose(ry - TIP_LOAD, 0.0, atol=1e-6)  # sum Fy = 0
    assert_allclose(mz - TIP_LOAD * LENGTH, 0.0, atol=1e-6)  # sum Mz about node 1 = 0
