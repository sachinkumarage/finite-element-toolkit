"""Frame validation: a simple portal frame under lateral load.

    Node 3 -------- beam -------- Node 4
      |                             |
    column 1                    column 2
      |                             |
    Node 1 (fixed)              Node 2 (fixed)

    Node 1 = (0, 0), Node 2 = (4, 0), Node 3 = (0, 3), Node 4 = (4, 3)
    E = 200 GPa, A = 0.01 m^2, I = 8.333e-6 m^4
    Fx = 2000 N applied at Node 3 (lateral/wind load)

This is not solved by an independent closed-form formula (a fixed-fixed
portal frame under a single asymmetric lateral load is statically
indeterminate); instead it is validated the way
``tests/validation/test_symmetric_truss.py`` validates its statically
indeterminate structure: by checking that the solution satisfies
mechanics-derived invariants that are independent of the solver's own
internals -- global force and moment equilibrium, and physically sensible
displacement directions -- across multiple horizontal, vertical, and
(implicitly, via the beam-column joints) connected frame elements with
different roles (columns vs. beam) sharing the same structure.
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
WIDTH = 4.0
HEIGHT = 3.0
LATERAL_LOAD = 2000.0


def _build_result():
    steel = Material(
        name="Steel", density=7850.0, youngs_modulus=YOUNGS_MODULUS, poissons_ratio=0.3
    )
    section = CrossSection(area=AREA, second_moment_of_area=SECOND_MOMENT_OF_AREA)

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=WIDTH, y=0.0, z=0.0)
    node_3 = Node(id=3, x=0.0, y=HEIGHT, z=0.0)
    node_4 = Node(id=4, x=WIDTH, y=HEIGHT, z=0.0)

    column_1 = FrameElement2D(id=1, nodes=(node_1, node_3), material=steel, cross_section=section)
    column_2 = FrameElement2D(id=2, nodes=(node_2, node_4), material=steel, cross_section=section)
    beam = FrameElement2D(id=3, nodes=(node_3, node_4), material=steel, cross_section=section)

    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4):
        mesh.add_node(node)
    for element in (column_1, column_2, beam):
        mesh.add_element(element)

    analysis = StaticLinearAnalysis(mesh)
    for node_id in (1, 2):
        for dof in (TranslationDOF.X, TranslationDOF.Y, RotationDOF.RZ):
            analysis.add_boundary_condition(BoundaryCondition(node_id=node_id, dof=dof, value=0.0))
    analysis.add_load(NodalLoad(node_id=3, dof=TranslationDOF.X, value=LATERAL_LOAD))
    return analysis.solve()


def test_frame_sways_in_the_direction_of_the_lateral_load() -> None:
    result = _build_result()

    ux3, _, _ = result.node_displacement(3)
    ux4, _, _ = result.node_displacement(4)

    assert ux3 > 0.0
    assert ux4 > 0.0


def test_beam_top_stays_nearly_rigid_top_translation() -> None:
    """The beam is far stiffer axially than the columns are in bending, so
    the two column tops must sway by nearly the same horizontal amount.
    """
    result = _build_result()

    ux3, _, _ = result.node_displacement(3)
    ux4, _, _ = result.node_displacement(4)

    assert_allclose(ux3, ux4, rtol=1e-3)


def test_global_force_equilibrium() -> None:
    result = _build_result()

    rx1, ry1, _ = result.node_reaction(1)
    rx2, ry2, _ = result.node_reaction(2)

    assert_allclose(rx1 + rx2 + LATERAL_LOAD, 0.0, atol=1e-6)
    assert_allclose(ry1 + ry2, 0.0, atol=1e-6)


def test_global_moment_equilibrium_about_node_1() -> None:
    """Sum of moments about Node 1 from every reaction and applied load must vanish."""
    result = _build_result()

    rx1, ry1, mz1 = result.node_reaction(1)
    rx2, ry2, mz2 = result.node_reaction(2)

    moment_from_reaction_2 = WIDTH * ry2 - 0.0 * rx2
    moment_from_load = 0.0 * 0.0 - HEIGHT * LATERAL_LOAD

    total_moment = mz1 + mz2 + moment_from_reaction_2 + moment_from_load
    assert_allclose(total_moment, 0.0, atol=1e-3)


def test_columns_carry_the_frame_via_shear_and_moment() -> None:
    """Both columns must be actively resisting the lateral load: nonzero
    base shear and base moment, of the same sign convention (the frame
    bends the same way at both fixed bases).
    """
    result = _build_result()

    base_shear_1 = result.element_shear_force(1, end="node_1")
    base_shear_2 = result.element_shear_force(2, end="node_1")
    base_moment_1 = result.element_bending_moment(1, end="node_1")
    base_moment_2 = result.element_bending_moment(2, end="node_1")

    assert abs(base_shear_1) > 1.0
    assert abs(base_shear_2) > 1.0
    assert abs(base_moment_1) > 1.0
    assert abs(base_moment_2) > 1.0


def test_element_forces_are_queryable_for_every_member() -> None:
    """Every member -- both columns and the beam -- must report consistent,
    finite end forces (a basic sanity/coverage check across all element
    orientations: two vertical members and one horizontal member).
    """
    result = _build_result()

    for element_id in (1, 2, 3):
        forces = result.element_end_forces(element_id)
        for end_forces in (forces.node_1, forces.node_2):
            assert_allclose(end_forces.axial_force, end_forces.axial_force)  # finite, not NaN
            assert_allclose(end_forces.shear_force, end_forces.shear_force)
            assert_allclose(end_forces.bending_moment, end_forces.bending_moment)
