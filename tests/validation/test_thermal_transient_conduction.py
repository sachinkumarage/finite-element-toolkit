"""Validation: transient heat conduction via Backward Euler (spec sections 11, 16).

Backward Euler is unconditionally stable: no matter how large the time
step, the solution cannot blow up, and -- for a problem whose boundary
conditions are held fixed -- taking a large enough total simulated time
must converge to exactly the same steady-state result a
:class:`~femtoolkit.thermal.thermal_analysis.SteadyStateThermalAnalysis`
would give directly. This file verifies that connection explicitly (the
transient solver's long-time limit *is* the steady-state solver), plus
the initial condition, monotonic heating/cooling behavior, and stability
under a very large time step.
"""

import numpy as np
import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import Material
from femtoolkit.mesh import Node
from femtoolkit.mesh.bar_element import BarElement
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.sections import CrossSection
from femtoolkit.thermal import (
    PrescribedTemperature,
    SteadyStateThermalAnalysis,
    ThermalMaterial,
    TransientThermalAnalysis,
)

_CONDUCTIVITY = 50.0
_DENSITY = 7850.0
_SPECIFIC_HEAT = 460.0
_LENGTH = 2.0
_T0 = 373.15
_T_L = 293.15


def _build_bar_mesh(num_elements: int) -> tuple[Mesh, dict[int, ThermalMaterial], list[Node]]:
    mechanical_material = Material(
        name="steel", density=_DENSITY, youngs_modulus=200e9, poissons_ratio=0.3
    )
    thermal_material = ThermalMaterial(
        thermal_conductivity=_CONDUCTIVITY, density=_DENSITY, specific_heat=_SPECIFIC_HEAT
    )
    x_coords = np.linspace(0.0, _LENGTH, num_elements + 1)
    nodes = [Node(id=i + 1, x=float(x), y=0.0, z=0.0) for i, x in enumerate(x_coords)]

    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)

    materials: dict[int, ThermalMaterial] = {}
    for i in range(num_elements):
        element = BarElement(
            id=i + 1,
            nodes=(nodes[i], nodes[i + 1]),
            material=mechanical_material,
            cross_section=CrossSection(area=0.01),
        )
        mesh.add_element(element)
        materials[element.id] = thermal_material

    return mesh, materials, nodes


def test_initial_temperature_is_recorded_as_the_first_step() -> None:
    mesh, materials, nodes = _build_bar_mesh(4)
    analysis = TransientThermalAnalysis(
        mesh, materials, time_step=10.0, num_steps=5, initial_temperature=300.0
    )
    analysis.add_boundary_condition(PrescribedTemperature(nodes[0].id, _T0))
    analysis.add_boundary_condition(PrescribedTemperature(nodes[-1].id, _T_L))
    result = analysis.solve()

    assert result.times[0] == pytest.approx(0.0)
    for node in nodes[1:-1]:
        assert result.node_temperature(node.id, step=0) == pytest.approx(300.0)
    # Boundary nodes start at their prescribed value, not the bulk initial temperature.
    assert result.node_temperature(nodes[0].id, step=0) == pytest.approx(_T0)
    assert result.node_temperature(nodes[-1].id, step=0) == pytest.approx(_T_L)


def test_per_node_initial_temperature_mapping() -> None:
    mesh, materials, nodes = _build_bar_mesh(2)
    initial = {node.id: 280.0 + 5.0 * i for i, node in enumerate(nodes)}
    analysis = TransientThermalAnalysis(
        mesh, materials, time_step=1.0, num_steps=1, initial_temperature=initial
    )
    analysis.add_boundary_condition(PrescribedTemperature(nodes[0].id, _T0))
    analysis.add_boundary_condition(PrescribedTemperature(nodes[-1].id, _T_L))
    result = analysis.solve()
    middle_node = nodes[1]
    assert result.node_temperature(middle_node.id, step=0) == pytest.approx(initial[middle_node.id])


def test_time_step_progression_and_history_length() -> None:
    mesh, materials, nodes = _build_bar_mesh(4)
    num_steps = 15
    analysis = TransientThermalAnalysis(
        mesh, materials, time_step=2.0, num_steps=num_steps, initial_temperature=293.15
    )
    analysis.add_boundary_condition(PrescribedTemperature(nodes[0].id, _T0))
    analysis.add_boundary_condition(PrescribedTemperature(nodes[-1].id, _T_L))
    result = analysis.solve()

    assert len(result.times) == num_steps + 1
    assert len(result.temperature_history) == num_steps + 1
    assert result.times[-1] == pytest.approx(2.0 * num_steps)
    assert np.all(np.diff(result.times) == pytest.approx(2.0))


def test_temperature_evolves_monotonically_toward_hotter_boundary() -> None:
    """A node starting cold, next to a hot fixed boundary, must heat up monotonically."""
    mesh, materials, nodes = _build_bar_mesh(6)
    analysis = TransientThermalAnalysis(
        mesh, materials, time_step=50.0, num_steps=40, initial_temperature=293.15
    )
    analysis.add_boundary_condition(PrescribedTemperature(nodes[0].id, _T0))
    analysis.add_boundary_condition(PrescribedTemperature(nodes[-1].id, 293.15))
    result = analysis.solve()

    history = result.node_temperature_history(nodes[1].id)
    assert np.all(np.diff(history) >= -1e-9)


def test_transient_converges_to_steady_state_solution_at_long_time() -> None:
    """Backward Euler's unconditional stability lets a single large time step reach
    (effectively) the true steady state directly -- verified against the steady-state
    solver's own result for the identical problem."""
    mesh, materials, nodes = _build_bar_mesh(4)

    steady_state_analysis = SteadyStateThermalAnalysis(mesh, materials)
    steady_state_analysis.add_boundary_condition(PrescribedTemperature(nodes[0].id, _T0))
    steady_state_analysis.add_boundary_condition(PrescribedTemperature(nodes[-1].id, _T_L))
    steady_state_result = steady_state_analysis.solve()

    transient_analysis = TransientThermalAnalysis(
        mesh, materials, time_step=1.0e7, num_steps=5, initial_temperature=293.15
    )
    transient_analysis.add_boundary_condition(PrescribedTemperature(nodes[0].id, _T0))
    transient_analysis.add_boundary_condition(PrescribedTemperature(nodes[-1].id, _T_L))
    transient_result = transient_analysis.solve()

    for node in nodes:
        assert transient_result.node_temperature(node.id, step=-1) == pytest.approx(
            steady_state_result.node_temperature(node.id), rel=1e-6
        )


def test_large_time_step_remains_stable_no_overshoot_beyond_boundary_values() -> None:
    """Unconditional stability: even a huge time step must not blow up or oscillate --
    every intermediate temperature must stay within the range spanned by the initial
    and boundary temperatures."""
    mesh, materials, nodes = _build_bar_mesh(4)
    analysis = TransientThermalAnalysis(
        mesh, materials, time_step=1.0e6, num_steps=10, initial_temperature=293.15
    )
    analysis.add_boundary_condition(PrescribedTemperature(nodes[0].id, _T0))
    analysis.add_boundary_condition(PrescribedTemperature(nodes[-1].id, _T_L))
    result = analysis.solve()

    low, high = min(293.15, _T0, _T_L), max(293.15, _T0, _T_L)
    for temperatures in result.temperature_history:
        assert np.all(temperatures >= low - 1e-6)
        assert np.all(temperatures <= high + 1e-6)


def test_rejects_non_positive_time_step() -> None:
    mesh, materials, _nodes = _build_bar_mesh(2)
    with pytest.raises(ValidationError):
        TransientThermalAnalysis(mesh, materials, time_step=0.0, num_steps=5)


def test_rejects_non_positive_num_steps() -> None:
    mesh, materials, _nodes = _build_bar_mesh(2)
    with pytest.raises(ValidationError):
        TransientThermalAnalysis(mesh, materials, time_step=1.0, num_steps=0)
