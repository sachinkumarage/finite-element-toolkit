"""Tests for AnalysisResult."""

import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import (
    BoundaryCondition,
    NodalLoad,
    RotationDOF,
    StaticLinearAnalysis,
    TranslationDOF,
)
from femtoolkit.exceptions import EntityNotFoundError, InvalidElementError, ValidationError
from femtoolkit.materials import LinearElastic2D, Material
from femtoolkit.mesh import BarElement, CSTElement2D, FrameElement2D, Mesh, Node, TrussElement2D
from femtoolkit.sections import CrossSection


@pytest.fixture
def single_bar_result():
    steel = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    section = CrossSection(area=0.01)
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)

    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    mesh.add_element(
        BarElement(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)
    )

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=0, value=0.0))
    analysis.add_load(NodalLoad(node_id=2, dof=0, value=1000.0))

    return analysis.solve(), steel, section


def test_result_displacement(single_bar_result) -> None:
    result, steel, section = single_bar_result

    assert_allclose(result.displacement(1), 0.0, atol=1e-12)
    assert_allclose(result.displacement(2), 1000.0 * 2.0 / (steel.youngs_modulus * section.area))


def test_result_reaction_matches_equilibrium(single_bar_result) -> None:
    result, _, _ = single_bar_result

    assert_allclose(result.reaction(1), -1000.0)
    assert_allclose(result.reaction(2), 0.0, atol=1e-6)


def test_result_element_strain(single_bar_result) -> None:
    result, steel, section = single_bar_result

    expected_strain = 1000.0 / (steel.youngs_modulus * section.area)
    assert_allclose(result.element_strain(1), expected_strain)


def test_result_element_stress(single_bar_result) -> None:
    result, steel, section = single_bar_result

    expected_stress = 1000.0 / section.area
    assert_allclose(result.element_stress(1), expected_stress)


def test_result_element_axial_force(single_bar_result) -> None:
    result, _, _ = single_bar_result

    assert_allclose(result.element_axial_force(1), 1000.0)


def test_result_displacement_unknown_node_raises(single_bar_result) -> None:
    result, _, _ = single_bar_result

    with pytest.raises(EntityNotFoundError):
        result.displacement(99)


def test_result_element_unknown_id_raises(single_bar_result) -> None:
    result, _, _ = single_bar_result

    with pytest.raises(EntityNotFoundError):
        result.element_stress(99)


# --- Frame element results (Version 5) ---

YOUNGS_MODULUS = 200e9
AREA = 0.01
SECOND_MOMENT_OF_AREA = 8.333e-6
EXTREME_FIBER_DISTANCE = 0.05
LENGTH = 2.0
TIP_LOAD = 1000.0


@pytest.fixture
def cantilever_result():
    """A single horizontal frame element, fixed at node 1, tip-loaded at node 2."""
    steel = Material(
        name="Steel", density=7850.0, youngs_modulus=YOUNGS_MODULUS, poissons_ratio=0.3
    )
    section = CrossSection(
        area=AREA,
        second_moment_of_area=SECOND_MOMENT_OF_AREA,
        extreme_fiber_distance=EXTREME_FIBER_DISTANCE,
    )
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=LENGTH, y=0.0, z=0.0)

    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    mesh.add_element(
        FrameElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)
    )

    analysis = StaticLinearAnalysis(mesh)
    for dof in (TranslationDOF.X, TranslationDOF.Y, RotationDOF.RZ):
        analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=dof, value=0.0))
    analysis.add_load(NodalLoad(node_id=2, dof=TranslationDOF.Y, value=-TIP_LOAD))

    return analysis.solve()


def test_node_displacement_returns_three_components_for_frame_analysis(cantilever_result) -> None:
    ux, uy, rz = cantilever_result.node_displacement(2)

    assert_allclose(ux, 0.0, atol=1e-12)
    assert uy < 0.0  # tip deflects downward under the downward load
    assert rz < 0.0  # tip rotates under the tip load


def test_node_reaction_returns_three_components_for_frame_analysis(cantilever_result) -> None:
    rx, ry, mz = cantilever_result.node_reaction(1)

    assert_allclose(rx, 0.0, atol=1e-6)
    assert_allclose(ry, TIP_LOAD, rtol=1e-9)
    assert_allclose(mz, TIP_LOAD * LENGTH, rtol=1e-9)


def test_node_displacement_still_returns_two_components_for_truss_analysis() -> None:
    """Version 4 behavior must be unchanged: a 2D truss analysis still returns (ux, uy)."""
    steel = Material(
        name="Steel", density=7850.0, youngs_modulus=YOUNGS_MODULUS, poissons_ratio=0.3
    )
    section = CrossSection(area=AREA)
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)

    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    mesh.add_element(
        TrussElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)
    )

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.X, value=0.0))
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.Y, value=0.0))
    analysis.add_boundary_condition(BoundaryCondition(node_id=2, dof=TranslationDOF.Y, value=0.0))
    analysis.add_load(NodalLoad(node_id=2, dof=TranslationDOF.X, value=1000.0))
    result = analysis.solve()

    displacement = result.node_displacement(2)
    assert len(displacement) == 2


def test_element_end_forces_cantilever(cantilever_result) -> None:
    forces = cantilever_result.element_end_forces(1)

    assert_allclose(forces.node_1.shear_force, TIP_LOAD, rtol=1e-9)
    assert_allclose(forces.node_1.bending_moment, TIP_LOAD * LENGTH, rtol=1e-9)
    assert_allclose(forces.node_2.shear_force, -TIP_LOAD, rtol=1e-9)
    assert_allclose(forces.node_2.bending_moment, 0.0, atol=1e-6)


def test_element_shear_force_default_end(cantilever_result) -> None:
    assert_allclose(cantilever_result.element_shear_force(1), TIP_LOAD, rtol=1e-9)
    assert_allclose(cantilever_result.element_shear_force(1, end="node_2"), -TIP_LOAD, rtol=1e-9)


def test_element_bending_moment_default_end(cantilever_result) -> None:
    assert_allclose(cantilever_result.element_bending_moment(1), TIP_LOAD * LENGTH, rtol=1e-9)
    assert_allclose(cantilever_result.element_bending_moment(1, end="node_2"), 0.0, atol=1e-6)


def test_element_bending_stress(cantilever_result) -> None:
    expected = (TIP_LOAD * LENGTH) * EXTREME_FIBER_DISTANCE / SECOND_MOMENT_OF_AREA
    assert_allclose(cantilever_result.element_bending_stress(1), expected, rtol=1e-9)


def test_element_bending_stress_without_extreme_fiber_distance_raises() -> None:
    steel = Material(
        name="Steel", density=7850.0, youngs_modulus=YOUNGS_MODULUS, poissons_ratio=0.3
    )
    section = CrossSection(area=AREA, second_moment_of_area=SECOND_MOMENT_OF_AREA)
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=LENGTH, y=0.0, z=0.0)

    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    mesh.add_element(
        FrameElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)
    )

    analysis = StaticLinearAnalysis(mesh)
    for dof in (TranslationDOF.X, TranslationDOF.Y, RotationDOF.RZ):
        analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=dof, value=0.0))
    analysis.add_load(NodalLoad(node_id=2, dof=TranslationDOF.Y, value=-TIP_LOAD))
    result = analysis.solve()

    with pytest.raises(ValidationError):
        result.element_bending_stress(1)


def test_element_end_forces_rejects_non_frame_element(single_bar_result) -> None:
    result, _, _ = single_bar_result

    with pytest.raises(InvalidElementError):
        result.element_end_forces(1)


def test_element_shear_force_rejects_non_frame_element(single_bar_result) -> None:
    result, _, _ = single_bar_result

    with pytest.raises(InvalidElementError):
        result.element_shear_force(1)


# --- Continuum element results (Version 6) ---


@pytest.fixture
def cst_result():
    """A single CST triangle with an exact constant-strain displacement
    field prescribed directly as the boundary conditions, so the result
    is analytically known without needing to hand-derive a load case.
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
    # Prescribe every DOF directly: u = 0.002*x, v = 0 -> uniaxial strain.
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.X, value=0.0))
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.Y, value=0.0))
    analysis.add_boundary_condition(BoundaryCondition(node_id=2, dof=TranslationDOF.X, value=0.002))
    analysis.add_boundary_condition(BoundaryCondition(node_id=2, dof=TranslationDOF.Y, value=0.0))
    analysis.add_boundary_condition(BoundaryCondition(node_id=3, dof=TranslationDOF.X, value=0.0))
    analysis.add_boundary_condition(BoundaryCondition(node_id=3, dof=TranslationDOF.Y, value=0.0))

    return analysis.solve(), material


def test_element_strain_returns_vector_for_continuum_element(cst_result) -> None:
    result, _ = cst_result

    strain = result.element_strain(1)
    assert_allclose(strain, [0.002, 0.0, 0.0], atol=1e-15)


def test_element_stress_returns_vector_for_continuum_element(cst_result) -> None:
    result, material = cst_result

    stress = result.element_stress(1)
    strain = result.element_strain(1)
    assert_allclose(stress, material.constitutive_matrix @ strain)


def test_element_von_mises(cst_result) -> None:
    result, _ = cst_result

    sigma_x, sigma_y, tau_xy = result.element_stress(1)
    expected = (sigma_x**2 - sigma_x * sigma_y + sigma_y**2 + 3 * tau_xy**2) ** 0.5
    assert_allclose(result.element_von_mises(1), expected)


def test_element_principal_stresses(cst_result) -> None:
    result, _ = cst_result

    sigma_x, sigma_y, tau_xy = result.element_stress(1)
    sigma_1, sigma_2 = result.element_principal_stresses(1)

    assert_allclose(sigma_1 + sigma_2, sigma_x + sigma_y)
    assert sigma_1 >= sigma_2


def test_element_axial_force_rejects_continuum_element(cst_result) -> None:
    result, _ = cst_result

    with pytest.raises(InvalidElementError):
        result.element_axial_force(1)


def test_element_von_mises_rejects_non_continuum_element(single_bar_result) -> None:
    result, _, _ = single_bar_result

    with pytest.raises(InvalidElementError):
        result.element_von_mises(1)


def test_element_principal_stresses_rejects_non_continuum_element(single_bar_result) -> None:
    result, _, _ = single_bar_result

    with pytest.raises(InvalidElementError):
        result.element_principal_stresses(1)


def test_element_strain_still_returns_scalar_for_bar_element(single_bar_result) -> None:
    """Version 3 behavior must be unchanged: a bar element's strain is still a float."""
    result, _, _ = single_bar_result

    strain = result.element_strain(1)
    assert isinstance(strain, float)
