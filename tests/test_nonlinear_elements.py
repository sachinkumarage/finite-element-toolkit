"""Tests for CST/Q4 nonlinear internal force and tangent stiffness."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis.nonlinear_elements import (
    NONLINEAR_CAPABLE_ELEMENT_TYPES,
    NonlinearElementState,
    cst_internal_force_and_tangent,
    element_internal_force_and_tangent,
    initial_element_state,
    quad_internal_force_and_tangent,
)
from femtoolkit.exceptions import InvalidElementError, ValidationError
from femtoolkit.materials import (
    ElasticMaterialAdapter,
    ElasticPerfectlyPlasticMaterial1D,
    LinearElastic2D,
)
from femtoolkit.materials.material import Material
from femtoolkit.mesh.bar_element import BarElement
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.node import Node
from femtoolkit.mesh.quad_element import QuadElement2D
from femtoolkit.sections import CrossSection

LINEAR_MATERIAL = LinearElastic2D(
    youngs_modulus=210e9, poisson_ratio=0.3, formulation="plane_stress"
)


@pytest.fixture
def cst_triangle() -> CSTElement2D:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=0.0, y=1.0, z=0.0)
    return CSTElement2D(
        id=1, nodes=(node_1, node_2, node_3), material=LINEAR_MATERIAL, thickness=0.01
    )


@pytest.fixture
def quad_square() -> QuadElement2D:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=1.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=1.0, z=0.0)
    return QuadElement2D(
        id=1, nodes=(node_1, node_2, node_3, node_4), material=LINEAR_MATERIAL, thickness=0.01
    )


# --- NONLINEAR_CAPABLE_ELEMENT_TYPES / initial_element_state ---


def test_nonlinear_capable_element_types_includes_cst_and_quad(
    cst_triangle: CSTElement2D, quad_square: QuadElement2D
) -> None:
    assert isinstance(cst_triangle, NONLINEAR_CAPABLE_ELEMENT_TYPES)
    assert isinstance(quad_square, NONLINEAR_CAPABLE_ELEMENT_TYPES)


def test_initial_element_state_cst_has_one_state(cst_triangle: CSTElement2D) -> None:
    material = ElasticMaterialAdapter.from_linear_elastic_2d(LINEAR_MATERIAL)
    state = initial_element_state(cst_triangle, material)

    assert len(state.states) == 1
    assert_allclose(state.states[0].strain, np.zeros(3))


def test_initial_element_state_quad_has_four_independent_states(quad_square: QuadElement2D) -> None:
    material = ElasticMaterialAdapter.from_linear_elastic_2d(LINEAR_MATERIAL)
    state = initial_element_state(quad_square, material)

    assert len(state.states) == 4
    # Independent objects, not four references to the same instance.
    ids = {id(gauss_state) for gauss_state in state.states}
    assert len(ids) == 4


def test_initial_element_state_rejects_unsupported_element_type() -> None:
    steel = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    bar = BarElement(
        id=1, nodes=(node_1, node_2), material=steel, cross_section=CrossSection(area=0.01)
    )
    material = ElasticMaterialAdapter.from_linear_elastic_2d(LINEAR_MATERIAL)

    with pytest.raises(InvalidElementError):
        initial_element_state(bar, material)


def test_initial_element_state_rejects_scalar_strain_material_on_cst(
    cst_triangle: CSTElement2D,
) -> None:
    material_1d = ElasticPerfectlyPlasticMaterial1D(youngs_modulus=200e9, yield_stress=250e6)

    with pytest.raises(ValidationError):
        initial_element_state(cst_triangle, material_1d)


def test_initial_element_state_rejects_scalar_strain_material_on_quad(
    quad_square: QuadElement2D,
) -> None:
    material_1d = ElasticPerfectlyPlasticMaterial1D(youngs_modulus=200e9, yield_stress=250e6)

    with pytest.raises(ValidationError):
        initial_element_state(quad_square, material_1d)


# --- CST internal force / tangent stiffness ---


def test_cst_elastic_adapter_matches_linear_stiffness_formula(cst_triangle: CSTElement2D) -> None:
    material = ElasticMaterialAdapter.from_linear_elastic_2d(LINEAR_MATERIAL)
    committed_state = initial_element_state(cst_triangle, material)
    displacements = np.array([0.0, 0.0, 0.001, 0.0002, -0.0003, 0.0008])

    f_int, k_t, trial_state = cst_internal_force_and_tangent(
        cst_triangle, material, displacements, committed_state
    )

    assert_allclose(f_int, cst_triangle.stiffness_matrix @ displacements, atol=1e-6)
    assert_allclose(k_t, cst_triangle.stiffness_matrix, atol=1e-6)
    assert len(trial_state.states) == 1


def test_cst_internal_force_zero_at_zero_displacement(cst_triangle: CSTElement2D) -> None:
    material = ElasticMaterialAdapter.from_linear_elastic_2d(LINEAR_MATERIAL)
    committed_state = initial_element_state(cst_triangle, material)
    displacements = np.zeros(6)

    f_int, _, _ = cst_internal_force_and_tangent(
        cst_triangle, material, displacements, committed_state
    )

    assert_allclose(f_int, np.zeros(6), atol=1e-12)


# --- Q4 internal force / tangent stiffness ---


def test_quad_elastic_adapter_matches_linear_stiffness_formula(quad_square: QuadElement2D) -> None:
    material = ElasticMaterialAdapter.from_linear_elastic_2d(LINEAR_MATERIAL)
    committed_state = initial_element_state(quad_square, material)
    displacements = np.array([0.0, 0.0, 0.001, 0.0, 0.0011, 0.0009, 0.0, 0.0007])

    f_int, k_t, trial_state = quad_internal_force_and_tangent(
        quad_square, material, displacements, committed_state
    )

    assert_allclose(f_int, quad_square.stiffness_matrix @ displacements, atol=1e-6)
    assert_allclose(k_t, quad_square.stiffness_matrix, atol=1e-6)
    assert len(trial_state.states) == 4


def test_quad_gauss_points_maintain_independent_state(quad_square: QuadElement2D) -> None:
    """A non-uniform displacement field (bending-like) should produce
    different strain/stress at different Gauss points -- confirming
    per-point evaluation rather than one shared/averaged state.
    """
    material = ElasticMaterialAdapter.from_linear_elastic_2d(LINEAR_MATERIAL)
    committed_state = initial_element_state(quad_square, material)
    # Pure bending-like displacement: top edge moves right, bottom fixed.
    displacements = np.array([0.0, 0.0, 0.0, 0.0, 0.002, 0.0, 0.001, 0.0])

    _, _, trial_state = quad_internal_force_and_tangent(
        quad_square, material, displacements, committed_state
    )

    stresses = [gauss_state.stress for gauss_state in trial_state.states]
    # Not every Gauss point should have exactly the same stress vector.
    assert not all(np.allclose(stresses[0], stress) for stress in stresses[1:])


def test_quad_committed_state_not_mutated_by_trial_evaluation(quad_square: QuadElement2D) -> None:
    material = ElasticMaterialAdapter.from_linear_elastic_2d(LINEAR_MATERIAL)
    committed_state = initial_element_state(quad_square, material)
    displacements = np.array([0.0, 0.0, 0.001, 0.0, 0.0011, 0.0009, 0.0, 0.0007])

    quad_internal_force_and_tangent(quad_square, material, displacements, committed_state)

    for gauss_state in committed_state.states:
        assert_allclose(gauss_state.strain, np.zeros(3))


# --- element_internal_force_and_tangent dispatcher ---


def test_dispatcher_routes_cst(cst_triangle: CSTElement2D) -> None:
    material = ElasticMaterialAdapter.from_linear_elastic_2d(LINEAR_MATERIAL)
    committed_state = initial_element_state(cst_triangle, material)
    displacements = np.zeros(6)

    f_int, k_t, _ = element_internal_force_and_tangent(
        cst_triangle, material, displacements, committed_state
    )
    assert f_int.shape == (6,)
    assert k_t.shape == (6, 6)


def test_dispatcher_routes_quad(quad_square: QuadElement2D) -> None:
    material = ElasticMaterialAdapter.from_linear_elastic_2d(LINEAR_MATERIAL)
    committed_state = initial_element_state(quad_square, material)
    displacements = np.zeros(8)

    f_int, k_t, _ = element_internal_force_and_tangent(
        quad_square, material, displacements, committed_state
    )
    assert f_int.shape == (8,)
    assert k_t.shape == (8, 8)


def test_dispatcher_rejects_unsupported_element_type() -> None:
    steel = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    bar = BarElement(
        id=1, nodes=(node_1, node_2), material=steel, cross_section=CrossSection(area=0.01)
    )
    material = ElasticMaterialAdapter.from_linear_elastic_2d(LINEAR_MATERIAL)
    state = NonlinearElementState(states=())

    with pytest.raises(InvalidElementError):
        element_internal_force_and_tangent(bar, material, [], state)
