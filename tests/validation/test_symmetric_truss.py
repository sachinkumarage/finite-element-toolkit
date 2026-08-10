"""Validation Case 3: symmetric truss regression test.

Geometry (isoceles triangle, symmetric about the vertical axis through Node 3):

          Node 3 (0, 1)
           /\\
          /  \\
         /    \\
        /      \\
Node 1 ---------- Node 2
 (-1,0)          (1,0)

Both Node 1 and Node 2 are fully pinned (ux=uy=0), making the structure
statically indeterminate -- not solvable by hand with the method of
joints alone. This is not an analytical-solution test; it is a
*consistency* test: symmetric geometry, symmetric supports, symmetric
material properties, and a symmetric (on-axis) load must produce
mirror-symmetric results. Any asymmetry in the result would indicate a
bug in geometry, assembly, or the solver.
"""

from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import Material
from femtoolkit.mesh import Mesh, Node, TrussElement2D
from femtoolkit.sections import CrossSection

APPLIED_LOAD = 1000.0


def _build_result():
    steel = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    section = CrossSection(area=0.001)

    node_1 = Node(id=1, x=-1.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=0.0, y=1.0, z=0.0)

    base = TrussElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)
    left = TrussElement2D(id=2, nodes=(node_1, node_3), material=steel, cross_section=section)
    right = TrussElement2D(id=3, nodes=(node_2, node_3), material=steel, cross_section=section)

    mesh = Mesh()
    for node in (node_1, node_2, node_3):
        mesh.add_node(node)
    for element in (base, left, right):
        mesh.add_element(element)

    analysis = StaticLinearAnalysis(mesh)
    for node_id in (1, 2):
        analysis.add_boundary_condition(
            BoundaryCondition(node_id=node_id, dof=TranslationDOF.X, value=0.0)
        )
        analysis.add_boundary_condition(
            BoundaryCondition(node_id=node_id, dof=TranslationDOF.Y, value=0.0)
        )
    analysis.add_load(NodalLoad(node_id=3, dof=TranslationDOF.Y, value=-APPLIED_LOAD))
    return analysis.solve()


def test_symmetric_truss_apex_has_no_horizontal_displacement() -> None:
    """A symmetric structure under an on-axis load must not deflect sideways."""
    result = _build_result()

    ux3, uy3 = result.node_displacement(3)
    assert_allclose(ux3, 0.0, atol=1e-12)
    assert uy3 < 0.0  # moves downward under the downward load


def test_symmetric_truss_mirrored_reactions() -> None:
    result = _build_result()

    rx1, ry1 = result.node_reaction(1)
    rx2, ry2 = result.node_reaction(2)

    assert_allclose(rx1, -rx2, rtol=1e-9)  # mirrored horizontal reactions
    assert_allclose(ry1, ry2, rtol=1e-9)  # equal vertical reactions
    assert_allclose(ry1, APPLIED_LOAD / 2, rtol=1e-9)  # load split evenly


def test_symmetric_truss_mirrored_member_forces() -> None:
    result = _build_result()

    assert_allclose(
        result.element_axial_force(2), result.element_axial_force(3), rtol=1e-9
    )  # left and right diagonals carry equal force


def test_symmetric_truss_global_equilibrium() -> None:
    result = _build_result()

    rx1, ry1 = result.node_reaction(1)
    rx2, ry2 = result.node_reaction(2)

    assert_allclose(rx1 + rx2, 0.0, atol=1e-6)
    assert_allclose(ry1 + ry2 - APPLIED_LOAD, 0.0, atol=1e-6)
