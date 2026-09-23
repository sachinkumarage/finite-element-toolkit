"""Tests for femtoolkit.verification.checks (Version 29)."""

from __future__ import annotations

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.analysis.system import build_force_vector
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh.generator import create_quad_mesh
from femtoolkit.thermal import PrescribedTemperature, SteadyStateThermalAnalysis, ThermalMaterial
from femtoolkit.verification.checks import check_force_equilibrium, check_thermal_energy_balance
from femtoolkit.verification.status import VerificationStatus
from femtoolkit.verification.tolerance import Tolerance

X, Y = TranslationDOF.X, TranslationDOF.Y


def _cantilever():
    material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    mesh = create_quad_mesh(width=4.0, height=1.0, nx=10, ny=3, material=material, thickness=0.02)
    analysis = StaticLinearAnalysis(mesh)
    left_nodes = [n for n in mesh.nodes if n.x == 0.0]
    right_nodes = [n for n in mesh.nodes if n.x == 4.0 and n.y == 0.0]
    for node in left_nodes:
        analysis.add_boundary_condition(BoundaryCondition(node.id, X, 0.0))
        analysis.add_boundary_condition(BoundaryCondition(node.id, Y, 0.0))
    loads = [NodalLoad(node.id, Y, -10000.0) for node in right_nodes]
    for load in loads:
        analysis.add_load(load)
    return analysis, loads


def test_force_equilibrium_passes_for_correctly_solved_cantilever() -> None:
    analysis, loads = _cantilever()
    result = analysis.solve()
    forces = build_force_vector(result.dof_map, loads)

    check = check_force_equilibrium(result.dof_map, forces, result.reactions)

    assert check.status is VerificationStatus.PASS
    assert len(check.components) == 2  # X, Y (QuadElement2D has dofs_per_node=2)


def test_force_equilibrium_reports_zero_applied_load_in_unloaded_direction() -> None:
    analysis, loads = _cantilever()
    result = analysis.solve()
    forces = build_force_vector(result.dof_map, loads)

    check = check_force_equilibrium(result.dof_map, forces, result.reactions)

    x_component = next(c for c in check.components if c.label == "X")
    assert x_component.applied_total == 0.0


def test_force_equilibrium_fails_on_artificially_corrupted_reactions() -> None:
    analysis, loads = _cantilever()
    result = analysis.solve()
    forces = build_force_vector(result.dof_map, loads)
    corrupted_reactions = result.reactions.copy()
    corrupted_reactions[0] += 1e6  # inject a large, obviously unbalanced force

    check = check_force_equilibrium(result.dof_map, forces, corrupted_reactions)

    assert check.status is VerificationStatus.FAIL


def test_force_equilibrium_scales_with_load_magnitude_via_relative_tolerance() -> None:
    """A large applied load should not require an unreasonably tight absolute tolerance."""
    analysis, loads = _cantilever()
    result = analysis.solve()
    forces = build_force_vector(result.dof_map, loads)

    loose_tolerance = Tolerance(absolute=1e-3, relative=1e-6)
    check = check_force_equilibrium(
        result.dof_map, forces, result.reactions, tolerance=loose_tolerance
    )
    assert check.status is VerificationStatus.PASS


def _thermal_plate():
    dummy = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    mesh = create_quad_mesh(width=1.0, height=1.0, nx=6, ny=6, material=dummy, thickness=0.01)
    thermal_material = ThermalMaterial(
        thermal_conductivity=45.0, density=7850.0, specific_heat=460.0
    )
    materials = {e.id: thermal_material for e in mesh.elements}
    analysis = SteadyStateThermalAnalysis(mesh, materials)
    for node in mesh.nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(PrescribedTemperature(node.id, 373.15))
        elif node.x == 1.0:
            analysis.add_boundary_condition(PrescribedTemperature(node.id, 293.15))
    return analysis


def test_thermal_energy_balance_passes_for_pure_conduction() -> None:
    analysis = _thermal_plate()
    result = analysis.solve()

    check = check_thermal_energy_balance(analysis, result)

    assert check.status is VerificationStatus.PASS
    assert check.components == []


def test_thermal_energy_balance_near_zero_supply_does_not_spuriously_fail() -> None:
    """A model with no explicit heat source has q_supplied == 0; a tiny
    floating-point q_removed must not be reported as a large relative failure."""
    analysis = _thermal_plate()
    result = analysis.solve()

    check = check_thermal_energy_balance(analysis, result)

    assert abs(check.metadata["q_supplied"]) < 1e-6
    assert check.status is VerificationStatus.PASS


def test_equilibrium_check_result_message_mentions_status() -> None:
    analysis, loads = _cantilever()
    result = analysis.solve()
    forces = build_force_vector(result.dof_map, loads)
    check = check_force_equilibrium(result.dof_map, forces, result.reactions)
    assert "PASS" in check.message
