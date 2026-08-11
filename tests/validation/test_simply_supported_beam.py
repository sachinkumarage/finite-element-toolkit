"""Validation Case 4: simply supported beam with a midspan point load.

A point load can only be applied at a node, so the beam is meshed with
two frame elements meeting at midspan, where the load is applied:

    Pin                Roller
    o--------- Node 2 ---------o
    Node 1     L/2      L/2    Node 3
    (ux=uy=0)  Fy = -1000 N    (uy=0)

    Total span L = 4 m, E = 200 GPa, I = 8.333e-6 m^4

Analytical solution (Euler-Bernoulli beam theory, point load at midspan
of a simply supported beam):

    R1 = R2 = P / 2
    M_max = P * L / 4      (at midspan)
    delta_mid = P * L^3 / (48 * E * I)   (downward)

This does not require a distributed-load capability: the entire loading
is a single nodal force at the shared midspan node.
"""

from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import Material
from femtoolkit.mesh import FrameElement2D, Mesh, Node
from femtoolkit.sections import CrossSection

YOUNGS_MODULUS = 200e9
AREA = 0.01
SECOND_MOMENT_OF_AREA = 8.333e-6
SPAN = 4.0
APPLIED_LOAD = 1000.0

ANALYTICAL_REACTION = APPLIED_LOAD / 2
ANALYTICAL_MAX_MOMENT = APPLIED_LOAD * SPAN / 4
ANALYTICAL_MID_DEFLECTION = -(
    APPLIED_LOAD * SPAN**3 / (48 * YOUNGS_MODULUS * SECOND_MOMENT_OF_AREA)
)


def _build_result():
    steel = Material(
        name="Steel", density=7850.0, youngs_modulus=YOUNGS_MODULUS, poissons_ratio=0.3
    )
    section = CrossSection(area=AREA, second_moment_of_area=SECOND_MOMENT_OF_AREA)

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=SPAN / 2, y=0.0, z=0.0)
    node_3 = Node(id=3, x=SPAN, y=0.0, z=0.0)
    left = FrameElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)
    right = FrameElement2D(id=2, nodes=(node_2, node_3), material=steel, cross_section=section)

    mesh = Mesh()
    for node in (node_1, node_2, node_3):
        mesh.add_node(node)
    for element in (left, right):
        mesh.add_element(element)

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.X, value=0.0))
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.Y, value=0.0))
    analysis.add_boundary_condition(BoundaryCondition(node_id=3, dof=TranslationDOF.Y, value=0.0))
    analysis.add_load(NodalLoad(node_id=2, dof=TranslationDOF.Y, value=-APPLIED_LOAD))
    return analysis.solve()


def test_support_reactions_split_evenly() -> None:
    result = _build_result()

    rx1, ry1, _ = result.node_reaction(1)
    _, ry3, _ = result.node_reaction(3)

    assert_allclose(rx1, 0.0, atol=1e-6)
    assert_allclose(ry1, ANALYTICAL_REACTION, rtol=1e-9)
    assert_allclose(ry3, ANALYTICAL_REACTION, rtol=1e-9)


def test_midspan_deflection_matches_analytical_solution() -> None:
    result = _build_result()

    ux, uy, _ = result.node_displacement(2)
    assert_allclose(uy, ANALYTICAL_MID_DEFLECTION, rtol=1e-9)


def test_bending_moment_at_midspan_matches_analytical_solution() -> None:
    """Both elements meeting at the midspan node must report the same
    moment magnitude there (continuity of the internal bending moment).
    """
    result = _build_result()

    moment_from_left = result.element_bending_moment(1, end="node_2")
    moment_from_right = result.element_bending_moment(2, end="node_1")

    assert_allclose(abs(moment_from_left), ANALYTICAL_MAX_MOMENT, rtol=1e-9)
    assert_allclose(abs(moment_from_right), ANALYTICAL_MAX_MOMENT, rtol=1e-9)


def test_bending_moment_is_zero_at_the_simple_supports() -> None:
    """A simple support carries no moment: pins and rollers cannot restrain rotation."""
    result = _build_result()

    assert_allclose(result.element_bending_moment(1, end="node_1"), 0.0, atol=1e-6)
    assert_allclose(result.element_bending_moment(2, end="node_2"), 0.0, atol=1e-6)


def test_global_equilibrium() -> None:
    result = _build_result()

    rx1, ry1, _ = result.node_reaction(1)
    _, ry3, _ = result.node_reaction(3)

    assert_allclose(rx1, 0.0, atol=1e-6)
    assert_allclose(ry1 + ry3 - APPLIED_LOAD, 0.0, atol=1e-6)
