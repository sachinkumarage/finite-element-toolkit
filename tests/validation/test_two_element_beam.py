"""Validation Case 5: two-element cantilever beam (assembly and continuity).

The same cantilever validated with a single element in
``test_cantilever_tip_load.py`` is rebuilt here with the same total
length split across two frame elements meeting at an intermediate node:

    Fixed                                     Free
    o----------------- o -----------------o
    Node 1   L/2      Node 2    L/2       Node 3
   (ux=uy=rz=0)                            Fy = -1000 N

    Total length L = 2 m, E = 200 GPa, I = 8.333e-6 m^4

If assembly, DOF continuity between elements, and the element-force
recovery are all correct, the two-element solution must agree with the
single-element analytical cantilever solution at the tip:

    delta = P * L^3 / (3 * E * I)
    theta = P * L^2 / (2 * E * I)

and the fixed-end (node 1) moment and shear must equal the single-element
values, since a cantilever's shear and moment diagrams do not depend on
how finely it happens to be meshed for a tip-only load.
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
    node_2 = Node(id=2, x=LENGTH / 2, y=0.0, z=0.0)
    node_3 = Node(id=3, x=LENGTH, y=0.0, z=0.0)
    first = FrameElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)
    second = FrameElement2D(id=2, nodes=(node_2, node_3), material=steel, cross_section=section)

    mesh = Mesh()
    for node in (node_1, node_2, node_3):
        mesh.add_node(node)
    for element in (first, second):
        mesh.add_element(element)

    analysis = StaticLinearAnalysis(mesh)
    for dof in (TranslationDOF.X, TranslationDOF.Y, RotationDOF.RZ):
        analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=dof, value=0.0))
    analysis.add_load(NodalLoad(node_id=3, dof=TranslationDOF.Y, value=-TIP_LOAD))
    return analysis.solve()


def test_tip_deflection_matches_single_element_solution() -> None:
    result = _build_result()

    assert_allclose(
        result.displacement(3, TranslationDOF.Y), ANALYTICAL_TIP_DEFLECTION, rtol=1e-9
    )


def test_tip_rotation_matches_single_element_solution() -> None:
    result = _build_result()

    assert_allclose(result.displacement(3, RotationDOF.RZ), ANALYTICAL_TIP_ROTATION, rtol=1e-9)


def test_fixed_end_moment_and_shear_are_unchanged_by_mesh_refinement() -> None:
    result = _build_result()

    assert_allclose(result.element_bending_moment(1), ANALYTICAL_FIXED_END_MOMENT, rtol=1e-9)
    assert_allclose(result.element_shear_force(1), ANALYTICAL_FIXED_END_SHEAR, rtol=1e-9)


def test_shear_is_continuous_across_the_intermediate_node() -> None:
    """With no load applied at node 2, shear must be continuous between elements."""
    result = _build_result()

    shear_from_first = result.element_shear_force(1, end="node_2")
    shear_from_second = result.element_shear_force(2, end="node_1")

    assert_allclose(shear_from_first, -shear_from_second, rtol=1e-9)


def test_intermediate_node_has_no_reaction() -> None:
    """Node 2 carries no boundary condition, so its net nodal force must be zero."""
    result = _build_result()

    rx, ry, mz = result.node_reaction(2)
    assert_allclose(rx, 0.0, atol=1e-6)
    assert_allclose(ry, 0.0, atol=1e-6)
    assert_allclose(mz, 0.0, atol=1e-6)
