"""Validation: the nonlinear (Newton-Raphson) thermal solver (spec sections 9, 10, 18).

Radiation (``q ~ T^4``) and temperature-dependent convection
(``h = h(T)``) both make the steady-state/transient thermal system
nonlinear in the unknown temperature -- this file verifies the
Newton-Raphson machinery built for that case
(:func:`femtoolkit.thermal.thermal_analysis._solve_nonlinear_system`/
:func:`femtoolkit.thermal.thermal_analysis._nonlinear_surface_terms`):
correctness of the analytical tangent (cross-checked against a
finite-difference of the residual -- the standard way to validate a
hand-derived Newton-Raphson tangent), convergence itself, and that a
convection boundary condition wrapped in a trivially-constant callable
(forcing the nonlinear code path even though the underlying physics is
linear) reproduces the exact same answer the fast linear solve path
gives directly.
"""

import numpy as np
import pytest

from femtoolkit.analysis.dof import DOFMap
from femtoolkit.exceptions import NonlinearConvergenceError, ValidationError
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, Node
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.thermal import PrescribedTemperature, SteadyStateThermalAnalysis, ThermalMaterial
from femtoolkit.thermal.thermal_analysis import _nonlinear_surface_terms
from femtoolkit.thermal.thermal_boundary_conditions import (
    ConvectionBoundaryCondition,
    RadiationBoundaryCondition,
)
from femtoolkit.thermal.thermal_surfaces import ThermalSurface

_CONDUCTIVITY = 50.0


def _plate_mesh() -> tuple[Mesh, dict[int, ThermalMaterial]]:
    placeholder = LinearElastic2D(
        youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress"
    )
    n1 = Node(1, 0.0, 0.0, 0.0)
    n2 = Node(2, 1.0, 0.0, 0.0)
    n3 = Node(3, 1.0, 1.0, 0.0)
    n4 = Node(4, 0.0, 1.0, 0.0)
    t1 = CSTElement2D(id=1, nodes=(n1, n2, n3), material=placeholder, thickness=0.01)
    t2 = CSTElement2D(id=2, nodes=(n1, n3, n4), material=placeholder, thickness=0.01)
    mesh = Mesh()
    for node in (n1, n2, n3, n4):
        mesh.add_node(node)
    mesh.add_element(t1)
    mesh.add_element(t2)
    thermal_material = ThermalMaterial(
        thermal_conductivity=_CONDUCTIVITY, density=7850.0, specific_heat=460.0
    )
    return mesh, {1: thermal_material, 2: thermal_material}


def test_nonlinear_tangent_matches_finite_difference_of_residual() -> None:
    """The mandatory Newton-Raphson tangent validation (spec section 18)."""
    mesh, _materials = _plate_mesh()
    convection_edge = ThermalSurface(1, 1)
    radiation_edge = ThermalSurface(2, 1)
    convections = [
        ConvectionBoundaryCondition(
            surface=convection_edge,
            convection_coefficient=lambda t: 10.0 + 0.05 * t,
            ambient_temperature=300.0,
        )
    ]
    radiations = [
        RadiationBoundaryCondition(
            surface=radiation_edge, emissivity=0.8, surrounding_temperature=280.0
        )
    ]
    dof_map = DOFMap(node_ids=[node.id for node in mesh.nodes], dofs_per_node=1)

    rng = np.random.default_rng(0)
    temperatures = 300.0 + 50.0 * rng.random(dof_map.total_dofs)

    _residual, tangent = _nonlinear_surface_terms(
        mesh, convections, radiations, dof_map, temperatures, time=0.0
    )

    step = 1e-3
    finite_difference_tangent = np.zeros_like(tangent)
    for j in range(dof_map.total_dofs):
        plus = temperatures.copy()
        plus[j] += step
        minus = temperatures.copy()
        minus[j] -= step
        residual_plus, _ = _nonlinear_surface_terms(
            mesh, convections, radiations, dof_map, plus, time=0.0
        )
        residual_minus, _ = _nonlinear_surface_terms(
            mesh, convections, radiations, dof_map, minus, time=0.0
        )
        finite_difference_tangent[:, j] = -(residual_plus - residual_minus) / (2.0 * step)

    assert tangent == pytest.approx(finite_difference_tangent, abs=1e-6)


def test_constant_convection_via_callable_matches_direct_linear_solve() -> None:
    """A convection BC wrapped in a trivially-constant callable forces the Newton
    code path, but must reproduce the exact linear-solve answer."""
    mesh, materials = _plate_mesh()
    edge = ThermalSurface(1, 1)

    linear_analysis = SteadyStateThermalAnalysis(mesh, materials)
    linear_analysis.add_boundary_condition(PrescribedTemperature(1, 400.0))
    linear_analysis.add_boundary_condition(PrescribedTemperature(4, 400.0))
    linear_analysis.add_convection(
        ConvectionBoundaryCondition(
            surface=edge, convection_coefficient=25.0, ambient_temperature=300.0
        )
    )
    linear_result = linear_analysis.solve()

    nonlinear_analysis = SteadyStateThermalAnalysis(mesh, materials)
    nonlinear_analysis.add_boundary_condition(PrescribedTemperature(1, 400.0))
    nonlinear_analysis.add_boundary_condition(PrescribedTemperature(4, 400.0))
    nonlinear_analysis.add_convection(
        ConvectionBoundaryCondition(
            surface=edge, convection_coefficient=lambda _t: 25.0, ambient_temperature=300.0
        )
    )
    nonlinear_result = nonlinear_analysis.solve()

    for node in mesh.nodes:
        assert nonlinear_result.node_temperature(node.id) == pytest.approx(
            linear_result.node_temperature(node.id), rel=1e-8
        )


def test_radiation_alone_converges_and_cools_the_exposed_edge() -> None:
    mesh, materials = _plate_mesh()
    edge = ThermalSurface(1, 1)
    analysis = SteadyStateThermalAnalysis(mesh, materials)
    analysis.add_boundary_condition(PrescribedTemperature(1, 500.0))
    analysis.add_boundary_condition(PrescribedTemperature(4, 500.0))
    analysis.add_radiation(
        RadiationBoundaryCondition(surface=edge, emissivity=0.8, surrounding_temperature=300.0)
    )
    result = analysis.solve()

    for node_id in (2, 3):
        temperature = result.node_temperature(node_id)
        assert 300.0 < temperature < 500.0


def test_combined_convection_and_radiation_converges() -> None:
    mesh, materials = _plate_mesh()
    edge = ThermalSurface(1, 1)
    analysis = SteadyStateThermalAnalysis(mesh, materials)
    analysis.add_boundary_condition(PrescribedTemperature(1, 500.0))
    analysis.add_boundary_condition(PrescribedTemperature(4, 500.0))
    analysis.add_convection(
        ConvectionBoundaryCondition(
            surface=edge, convection_coefficient=20.0, ambient_temperature=300.0
        )
    )
    analysis.add_radiation(
        RadiationBoundaryCondition(surface=edge, emissivity=0.8, surrounding_temperature=300.0)
    )
    result = analysis.solve()

    for node_id in (2, 3):
        temperature = result.node_temperature(node_id)
        assert 300.0 < temperature < 500.0


def test_multiple_thermal_boundary_conditions_on_different_surfaces() -> None:
    """Prescribed temperature, convection, and radiation applied to different surfaces at once."""
    mesh, materials = _plate_mesh()
    convection_edge = ThermalSurface(1, 1)
    radiation_edge = ThermalSurface(2, 1)
    analysis = SteadyStateThermalAnalysis(mesh, materials)
    analysis.add_boundary_condition(PrescribedTemperature(1, 500.0))
    analysis.add_convection(
        ConvectionBoundaryCondition(
            surface=convection_edge, convection_coefficient=25.0, ambient_temperature=310.0
        )
    )
    analysis.add_radiation(
        RadiationBoundaryCondition(
            surface=radiation_edge, emissivity=0.7, surrounding_temperature=290.0
        )
    )
    result = analysis.solve()
    assert result.node_temperature(1) == pytest.approx(500.0)
    for node_id in (2, 3, 4):
        assert result.node_temperature(node_id) < 500.0


def test_temperature_dependent_convection_changes_the_result_relative_to_constant_h() -> None:
    """A genuinely T-dependent h must give a different answer than freezing h at one value."""
    mesh, materials = _plate_mesh()
    edge = ThermalSurface(1, 1)

    constant = SteadyStateThermalAnalysis(mesh, materials)
    constant.add_boundary_condition(PrescribedTemperature(1, 500.0))
    constant.add_boundary_condition(PrescribedTemperature(4, 500.0))
    constant.add_convection(
        ConvectionBoundaryCondition(
            surface=edge, convection_coefficient=10.0, ambient_temperature=300.0
        )
    )
    constant_result = constant.solve()

    variable = SteadyStateThermalAnalysis(mesh, materials)
    variable.add_boundary_condition(PrescribedTemperature(1, 500.0))
    variable.add_boundary_condition(PrescribedTemperature(4, 500.0))
    variable.add_convection(
        ConvectionBoundaryCondition(
            surface=edge,
            convection_coefficient=lambda t: 10.0 + 0.2 * (t - 300.0),
            ambient_temperature=300.0,
        )
    )
    variable_result = variable.solve()

    assert constant_result.node_temperature(2) != pytest.approx(variable_result.node_temperature(2))


def test_nonlinear_solve_raises_on_duplicate_prescribed_temperature() -> None:
    mesh, materials = _plate_mesh()
    edge = ThermalSurface(1, 1)
    analysis = SteadyStateThermalAnalysis(mesh, materials)
    analysis.add_boundary_condition(PrescribedTemperature(1, 400.0))
    analysis.add_boundary_condition(PrescribedTemperature(1, 410.0))
    analysis.add_radiation(
        RadiationBoundaryCondition(surface=edge, emissivity=0.8, surrounding_temperature=300.0)
    )
    with pytest.raises(ValidationError):
        analysis.solve()


def test_nonlinear_solve_raises_convergence_error_when_starved_of_iterations() -> None:
    mesh, materials = _plate_mesh()
    edge = ThermalSurface(1, 1)
    analysis = SteadyStateThermalAnalysis(mesh, materials, nonlinear_max_iterations=1)
    analysis.add_boundary_condition(PrescribedTemperature(1, 900.0))
    analysis.add_boundary_condition(PrescribedTemperature(4, 900.0))
    analysis.add_radiation(
        RadiationBoundaryCondition(surface=edge, emissivity=0.9, surrounding_temperature=250.0)
    )
    with pytest.raises(NonlinearConvergenceError):
        analysis.solve()


def test_rejects_non_positive_nonlinear_max_iterations() -> None:
    mesh, materials = _plate_mesh()
    with pytest.raises(ValidationError):
        SteadyStateThermalAnalysis(mesh, materials, nonlinear_max_iterations=0)


def test_rejects_non_positive_nonlinear_tolerance() -> None:
    mesh, materials = _plate_mesh()
    with pytest.raises(ValidationError):
        SteadyStateThermalAnalysis(mesh, materials, nonlinear_tolerance=0.0)
