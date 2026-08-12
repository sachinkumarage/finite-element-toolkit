"""Tests for StaticLinearAnalysis."""

import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.exceptions import (
    InsufficientConstraintsError,
    InvalidAnalysisError,
    InvalidElementError,
    SingularSystemError,
)
from femtoolkit.materials import LinearElastic2D, Material
from femtoolkit.mesh import BarElement, CSTElement2D, Element, FrameElement2D, Mesh, Node
from femtoolkit.sections import CrossSection


@pytest.fixture
def steel() -> Material:
    return Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)


@pytest.fixture
def section() -> CrossSection:
    return CrossSection(area=0.01)


def _single_bar_mesh(steel: Material, section: CrossSection) -> tuple[Mesh, Node, Node]:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)
    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    mesh.add_element(
        BarElement(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)
    )
    return mesh, node_1, node_2


def test_solve_single_bar(steel: Material, section: CrossSection) -> None:
    mesh, node_1, node_2 = _single_bar_mesh(steel, section)

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(node_id=node_1.id, dof=0, value=0.0))
    analysis.add_load(NodalLoad(node_id=node_2.id, dof=0, value=1000.0))
    result = analysis.solve()

    assert_allclose(result.displacement(node_2.id), 1000.0 * 2.0 / (200e9 * 0.01))


def test_solve_requires_boundary_conditions(steel: Material, section: CrossSection) -> None:
    mesh, node_1, node_2 = _single_bar_mesh(steel, section)

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_load(NodalLoad(node_id=node_2.id, dof=0, value=1000.0))

    with pytest.raises(InsufficientConstraintsError):
        analysis.solve()


def test_solve_detects_singular_system(steel: Material, section: CrossSection) -> None:
    """node_3 is disconnected from the constrained substructure: its DOF is
    a free mechanism, so the reduced global stiffness matrix is singular.
    """
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=4.0, y=0.0, z=0.0)
    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    mesh.add_node(node_3)
    mesh.add_element(
        BarElement(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)
    )
    # No element connects node_3 to the rest of the structure.

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(node_id=node_1.id, dof=0, value=0.0))
    analysis.add_load(NodalLoad(node_id=node_3.id, dof=0, value=1000.0))

    with pytest.raises(SingularSystemError):
        analysis.solve()


def test_solve_rejects_empty_mesh() -> None:
    analysis = StaticLinearAnalysis(Mesh())

    with pytest.raises(InvalidAnalysisError):
        analysis.solve()


def test_solve_rejects_mesh_with_nodes_but_no_elements() -> None:
    mesh = Mesh()
    mesh.add_node(Node(id=1, x=0.0, y=0.0, z=0.0))

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=0, value=0.0))

    with pytest.raises(InvalidAnalysisError):
        analysis.solve()


def test_solve_detects_singular_frame_system(steel: Material) -> None:
    """A frame element with only its axial DOF constrained is a rigid-body
    mechanism: it remains free to translate in Y and rotate, so the
    reduced global stiffness matrix is singular. This is the Version 5
    structural-instability check (see the analysis module docstring):
    :class:`StaticLinearAnalysis` detects it the same way for any element
    type, without a frame-specific code path.
    """
    section = CrossSection(area=0.01, second_moment_of_area=8.333e-6)
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)
    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    mesh.add_element(
        FrameElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)
    )

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.X, value=0.0))
    analysis.add_load(NodalLoad(node_id=2, dof=TranslationDOF.Y, value=1000.0))

    with pytest.raises(SingularSystemError):
        analysis.solve()


def test_solve_two_cst_elements_sharing_a_node() -> None:
    """A two-triangle mesh sharing an edge (two nodes) must assemble and
    solve through the exact same generic code path as bar/truss/frame
    meshes, without any CST-specific branch in StaticLinearAnalysis.
    """
    material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=1.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=1.0, z=0.0)

    lower = CSTElement2D(id=1, nodes=(node_1, node_2, node_3), material=material, thickness=0.01)
    upper = CSTElement2D(id=2, nodes=(node_1, node_3, node_4), material=material, thickness=0.01)

    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4):
        mesh.add_node(node)
    for element in (lower, upper):
        mesh.add_element(element)

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.X, value=0.0))
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.Y, value=0.0))
    analysis.add_boundary_condition(BoundaryCondition(node_id=4, dof=TranslationDOF.X, value=0.0))
    analysis.add_load(NodalLoad(node_id=2, dof=TranslationDOF.X, value=1000.0))
    analysis.add_load(NodalLoad(node_id=3, dof=TranslationDOF.X, value=1000.0))
    result = analysis.solve()

    # Node 1 is fully fixed; the mesh must deform without raising.
    ux3, _ = result.node_displacement(3)
    assert ux3 > 0.0


def test_solve_detects_singular_cst_system() -> None:
    """A single CST triangle constrained only in X is free to translate
    and rotate rigidly in Y -- the continuum analogue of the Version 4/5
    structural-instability checks.
    """
    material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=0.0, y=1.0, z=0.0)
    element = CSTElement2D(id=1, nodes=(node_1, node_2, node_3), material=material, thickness=0.01)

    mesh = Mesh()
    for node in (node_1, node_2, node_3):
        mesh.add_node(node)
    mesh.add_element(element)

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.X, value=0.0))
    analysis.add_load(NodalLoad(node_id=2, dof=TranslationDOF.Y, value=1000.0))

    with pytest.raises(SingularSystemError):
        analysis.solve()


def test_solve_rejects_non_bar_elements(steel: Material) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)
    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    mesh.add_element(Element(id=1, nodes=[node_1, node_2], material=steel))

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=0, value=0.0))

    with pytest.raises(InvalidElementError):
        analysis.solve()
