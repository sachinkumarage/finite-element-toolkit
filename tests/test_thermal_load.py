"""Tests for TemperatureLoad, thermal_load_to_nodal_loads, and thermal-corrected stress/strain."""

import math

import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, StaticLinearAnalysis, TranslationDOF
from femtoolkit.analysis.thermal_load import (
    TemperatureLoad,
    thermal_corrected_strain,
    thermal_corrected_stress,
    thermal_load_to_nodal_loads,
)
from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import CSTElement2D, Mesh, Node, QuadElement2D

ALPHA = 12e-6
DELTA_T = 100.0
THICKNESS = 0.01


@pytest.fixture
def material() -> LinearElastic2D:
    return LinearElastic2D(
        youngs_modulus=200e9,
        poisson_ratio=0.3,
        formulation="plane_stress",
        thermal_expansion_coefficient=ALPHA,
    )


@pytest.fixture
def material_without_alpha() -> LinearElastic2D:
    return LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")


@pytest.fixture
def cst_mesh(material: LinearElastic2D) -> Mesh:
    mesh = Mesh()
    mesh.add_node(Node(1, 0.0, 0.0, 0.0))
    mesh.add_node(Node(2, 1.0, 0.0, 0.0))
    mesh.add_node(Node(3, 0.0, 1.0, 0.0))
    mesh.add_element(
        CSTElement2D(id=1, nodes=(mesh.get_node(1), mesh.get_node(2), mesh.get_node(3)),
                      material=material, thickness=THICKNESS)
    )
    return mesh


# --- TemperatureLoad validation ---


def test_temperature_load_rejects_non_finite_delta() -> None:
    with pytest.raises(ValidationError):
        TemperatureLoad(delta_temperature=math.inf)


def test_temperature_load_stores_delta() -> None:
    load = TemperatureLoad(delta_temperature=50.0)
    assert load.delta_temperature == 50.0


# --- thermal_load_to_nodal_loads: requires alpha ---


def test_cst_thermal_requires_expansion_coefficient(
    material_without_alpha: LinearElastic2D,
) -> None:
    mesh = Mesh()
    mesh.add_node(Node(1, 0.0, 0.0, 0.0))
    mesh.add_node(Node(2, 1.0, 0.0, 0.0))
    mesh.add_node(Node(3, 0.0, 1.0, 0.0))
    mesh.add_element(
        CSTElement2D(id=1, nodes=(mesh.get_node(1), mesh.get_node(2), mesh.get_node(3)),
                      material=material_without_alpha, thickness=THICKNESS)
    )

    with pytest.raises(ValidationError):
        thermal_load_to_nodal_loads(mesh, TemperatureLoad(delta_temperature=DELTA_T))


def test_no_contributing_elements_returns_empty_list() -> None:
    assert thermal_load_to_nodal_loads(Mesh(), TemperatureLoad(delta_temperature=DELTA_T)) == []


# --- Free thermal expansion: displacement matches alpha*dT*length, zero mechanical stress ---


def _pin_and_roller_analysis(mesh: Mesh, loads) -> StaticLinearAnalysis:
    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(1, TranslationDOF.X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, TranslationDOF.Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(2, TranslationDOF.Y, 0.0))
    for load in loads:
        analysis.add_load(load)
    return analysis


def test_cst_free_thermal_expansion_matches_alpha_delta_t(cst_mesh: Mesh) -> None:
    thermal_load = TemperatureLoad(delta_temperature=DELTA_T)
    loads = thermal_load_to_nodal_loads(cst_mesh, thermal_load)
    result = _pin_and_roller_analysis(cst_mesh, loads).solve()

    assert_allclose(result.displacement(2, TranslationDOF.X), ALPHA * DELTA_T, rtol=1e-9)
    assert_allclose(result.displacement(3, TranslationDOF.Y), ALPHA * DELTA_T, rtol=1e-9)


def test_cst_free_thermal_expansion_has_zero_mechanical_stress(cst_mesh: Mesh) -> None:
    """Free expansion produces displacement, not stress: the raw
    element_stress() (total-strain-based, see the thermal_load module
    docstring) is spuriously large, but the thermal-corrected stress,
    which subtracts the free thermal eigenstrain, must be ~0.
    """
    thermal_load = TemperatureLoad(delta_temperature=DELTA_T)
    loads = thermal_load_to_nodal_loads(cst_mesh, thermal_load)
    result = _pin_and_roller_analysis(cst_mesh, loads).solve()
    element = cst_mesh.get_element(1)

    corrected = thermal_corrected_stress(result, element, thermal_load)
    assert_allclose(corrected, [0.0, 0.0, 0.0], atol=1e-3)

    corrected_strain = thermal_corrected_strain(result, element, thermal_load)
    assert_allclose(corrected_strain, [0.0, 0.0, 0.0], atol=1e-12)


def test_cst_fully_restrained_thermal_stress_matches_analytical(cst_mesh: Mesh) -> None:
    """A fully restrained element cannot expand at all: mechanical strain
    equals -epsilon_thermal exactly, giving an exact analytical thermal
    stress, sigma = -D @ epsilon_thermal.
    """
    thermal_load = TemperatureLoad(delta_temperature=DELTA_T)
    loads = thermal_load_to_nodal_loads(cst_mesh, thermal_load)

    analysis = StaticLinearAnalysis(cst_mesh)
    for node_id in (1, 2, 3):
        analysis.add_boundary_condition(BoundaryCondition(node_id, TranslationDOF.X, 0.0))
        analysis.add_boundary_condition(BoundaryCondition(node_id, TranslationDOF.Y, 0.0))
    for load in loads:
        analysis.add_load(load)
    result = analysis.solve()

    element = cst_mesh.get_element(1)
    material = element.material
    epsilon_thermal = ALPHA * DELTA_T
    expected = -(material.constitutive_matrix @ [epsilon_thermal, epsilon_thermal, 0.0])

    corrected = thermal_corrected_stress(result, element, thermal_load)
    assert_allclose(corrected, expected, rtol=1e-9)

    # Every displacement must be exactly zero (fully restrained).
    for node_id in (1, 2, 3):
        assert_allclose(result.node_displacement(node_id), (0.0, 0.0), atol=1e-12)


def test_quad_free_thermal_expansion_matches_alpha_delta_t(material: LinearElastic2D) -> None:
    mesh = Mesh()
    mesh.add_node(Node(1, 0.0, 0.0, 0.0))
    mesh.add_node(Node(2, 1.0, 0.0, 0.0))
    mesh.add_node(Node(3, 1.0, 1.0, 0.0))
    mesh.add_node(Node(4, 0.0, 1.0, 0.0))
    mesh.add_element(
        QuadElement2D(
            id=1,
            nodes=(mesh.get_node(1), mesh.get_node(2), mesh.get_node(3), mesh.get_node(4)),
            material=material,
            thickness=THICKNESS,
        )
    )

    thermal_load = TemperatureLoad(delta_temperature=DELTA_T)
    loads = thermal_load_to_nodal_loads(mesh, thermal_load)

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(1, TranslationDOF.X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, TranslationDOF.Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(2, TranslationDOF.Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, TranslationDOF.X, 0.0))
    for load in loads:
        analysis.add_load(load)
    result = analysis.solve()

    assert_allclose(result.displacement(2, TranslationDOF.X), ALPHA * DELTA_T, rtol=1e-9)
    assert_allclose(result.node_displacement(3), (ALPHA * DELTA_T, ALPHA * DELTA_T), rtol=1e-9)

    element = mesh.get_element(1)
    corrected = thermal_corrected_stress(result, element, thermal_load)
    assert_allclose(corrected, [0.0, 0.0, 0.0], atol=1e-3)
