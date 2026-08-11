"""Validation Case 1: axial bar regression via a horizontal frame element.

A single :class:`~femtoolkit.mesh.frame_element.FrameElement2D`, loaded
only along its axis, must reproduce the Version 3 fixed-free axial bar
solution exactly (its axial DOFs are, by construction, uncoupled from its
bending DOFs -- see ``test_frame_local_stiffness_axial_bending_uncoupled``
in ``tests/test_stiffness.py``). This proves the frame element correctly
contains the same axial stiffness as :class:`~femtoolkit.mesh.bar_element.BarElement`
and :class:`~femtoolkit.mesh.truss_element.TrussElement2D`, not a
different or approximate one.

Model:

    Node 1 (fixed: ux=uy=rz=0) -------- Node 2 (Fx = 1000 N)
                       Frame Element, L = 2 m

    E = 200 GPa, A = 0.01 m^2, I = 8.333e-6 m^4 (unused by this load case)

Analytical solution (identical to the Version 3 bar):

    u = F * L / (E * A)
    N = F
    sigma = F / A
    Rx = -F
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
APPLIED_FORCE = 1000.0

ANALYTICAL_DISPLACEMENT = APPLIED_FORCE * LENGTH / (YOUNGS_MODULUS * AREA)
ANALYTICAL_STRESS = APPLIED_FORCE / AREA


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
    analysis.add_load(NodalLoad(node_id=2, dof=TranslationDOF.X, value=APPLIED_FORCE))
    return analysis.solve()


def test_axial_displacement_matches_bar_solution() -> None:
    result = _build_result()

    assert_allclose(
        result.displacement(2, TranslationDOF.X), ANALYTICAL_DISPLACEMENT, rtol=1e-9
    )


def test_axial_force_matches_applied_load() -> None:
    result = _build_result()

    assert_allclose(result.element_axial_force(1), APPLIED_FORCE, rtol=1e-9)


def test_axial_stress_matches_bar_solution() -> None:
    result = _build_result()

    assert_allclose(result.element_stress(1), ANALYTICAL_STRESS, rtol=1e-9)


def test_reaction_matches_equilibrium() -> None:
    result = _build_result()

    rx, ry, mz = result.node_reaction(1)
    assert_allclose(rx, -APPLIED_FORCE, rtol=1e-9)
    assert_allclose(ry, 0.0, atol=1e-6)
    assert_allclose(mz, 0.0, atol=1e-6)


def test_no_bending_is_induced_by_a_purely_axial_load() -> None:
    """A frame element under a purely axial load must show zero shear and moment."""
    result = _build_result()

    forces = result.element_end_forces(1)
    assert_allclose(forces.node_1.shear_force, 0.0, atol=1e-6)
    assert_allclose(forces.node_1.bending_moment, 0.0, atol=1e-6)
    assert_allclose(forces.node_2.shear_force, 0.0, atol=1e-6)
    assert_allclose(forces.node_2.bending_moment, 0.0, atol=1e-6)
