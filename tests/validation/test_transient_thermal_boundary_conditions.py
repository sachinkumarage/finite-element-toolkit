"""Validation: transient convection/radiation boundary conditions (spec sections 12, 18).

Verifies that :class:`~femtoolkit.thermal.thermal_analysis.TransientThermalAnalysis`
re-resolves a time-dependent ambient temperature (a furnace heating
cycle) at each time step, that a nonlinear (temperature-dependent
convection, or radiation) boundary condition is handled correctly inside
the transient Backward-Euler loop, and that a transient run with
constant-coefficient convection converges to the same steady state the
convection benchmark (``test_convection_benchmark.py``) verifies
directly.
"""

import numpy as np
import pytest

from femtoolkit.analysis.dynamic_loads import StepLoad
from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.thermal import (
    PrescribedTemperature,
    SteadyStateThermalAnalysis,
    ThermalMaterial,
    TransientThermalAnalysis,
)
from femtoolkit.thermal.thermal_boundary_conditions import (
    ConvectionBoundaryCondition,
    RadiationBoundaryCondition,
)
from femtoolkit.thermal.thermal_surfaces import ThermalSurface

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]
_RIGHT_FACE_INDEX = 4


def _build_slab() -> tuple[Mesh, tuple[Node, ...]]:
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)
    return mesh, nodes


def test_transient_linear_convection_converges_to_the_same_steady_state() -> None:
    mesh, nodes = _build_slab()
    conductivity, coefficient, hot_temperature, ambient_temperature = 50.0, 10.0, 400.0, 300.0
    material = ThermalMaterial(
        thermal_conductivity=conductivity, density=7850.0, specific_heat=460.0
    )
    right_face = ThermalSurface(1, _RIGHT_FACE_INDEX)

    steady = SteadyStateThermalAnalysis(mesh, {1: material})
    for node in nodes:
        if node.x == 0.0:
            steady.add_boundary_condition(PrescribedTemperature(node.id, hot_temperature))
    steady.add_convection(
        ConvectionBoundaryCondition(
            surface=right_face,
            convection_coefficient=coefficient,
            ambient_temperature=ambient_temperature,
        )
    )
    steady_result = steady.solve()

    transient = TransientThermalAnalysis(
        mesh, {1: material}, time_step=1.0e7, num_steps=5, initial_temperature=350.0
    )
    for node in nodes:
        if node.x == 0.0:
            transient.add_boundary_condition(PrescribedTemperature(node.id, hot_temperature))
    transient.add_convection(
        ConvectionBoundaryCondition(
            surface=right_face,
            convection_coefficient=coefficient,
            ambient_temperature=ambient_temperature,
        )
    )
    transient_result = transient.solve()

    for node in nodes:
        assert transient_result.node_temperature(node.id, step=-1) == pytest.approx(
            steady_result.node_temperature(node.id), rel=1e-6
        )


def test_transient_nonlinear_convection_converges_to_the_same_steady_state() -> None:
    """A temperature-dependent h must drive the same transient/steady-state Newton solvers
    toward the same long-time limit."""
    mesh, nodes = _build_slab()
    conductivity, hot_temperature, ambient_temperature = 50.0, 400.0, 300.0

    def coefficient(temperature: float) -> float:
        return 10.0 + 0.02 * (temperature - 300.0)

    material = ThermalMaterial(
        thermal_conductivity=conductivity, density=7850.0, specific_heat=460.0
    )
    right_face = ThermalSurface(1, _RIGHT_FACE_INDEX)

    steady = SteadyStateThermalAnalysis(mesh, {1: material})
    for node in nodes:
        if node.x == 0.0:
            steady.add_boundary_condition(PrescribedTemperature(node.id, hot_temperature))
    steady.add_convection(
        ConvectionBoundaryCondition(
            surface=right_face,
            convection_coefficient=coefficient,
            ambient_temperature=ambient_temperature,
        )
    )
    steady_result = steady.solve()

    transient = TransientThermalAnalysis(
        mesh, {1: material}, time_step=1.0e7, num_steps=5, initial_temperature=350.0
    )
    for node in nodes:
        if node.x == 0.0:
            transient.add_boundary_condition(PrescribedTemperature(node.id, hot_temperature))
    transient.add_convection(
        ConvectionBoundaryCondition(
            surface=right_face,
            convection_coefficient=coefficient,
            ambient_temperature=ambient_temperature,
        )
    )
    transient_result = transient.solve()

    for node in nodes:
        assert transient_result.node_temperature(node.id, step=-1) == pytest.approx(
            steady_result.node_temperature(node.id), rel=1e-6
        )


def test_time_dependent_ambient_temperature_updates_each_step() -> None:
    """A furnace-cycle ambient temperature (a step change at t=50s) must only affect
    later time steps, not earlier ones."""
    mesh, nodes = _build_slab()
    material = ThermalMaterial(thermal_conductivity=50.0, density=7850.0, specific_heat=460.0)
    right_face = ThermalSurface(1, _RIGHT_FACE_INDEX)

    furnace = StepLoad(magnitude=800.0, step_time=50.0)
    analysis = TransientThermalAnalysis(
        mesh, {1: material}, time_step=10.0, num_steps=10, initial_temperature=300.0
    )
    for node in nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(PrescribedTemperature(node.id, 300.0))
    analysis.add_convection(
        ConvectionBoundaryCondition(
            surface=right_face, convection_coefficient=25.0, ambient_temperature=furnace
        )
    )
    result = analysis.solve()

    # Before t=50s, ambient is 0K (StepLoad's pre-step value) -- the right face must cool
    # below the (300K) Dirichlet face rather than being pulled toward 800K.
    for step, time in enumerate(result.times):
        if time < 50.0:
            assert result.node_temperature(2, step=step) <= 300.0 + 1e-6

    # After the furnace switches on, the exposed face must trend upward, toward 800K.
    history = result.node_temperature_history(2)
    late_steps = [i for i, t in enumerate(result.times) if t > 50.0]
    assert all(np.diff(history[late_steps]) >= -1e-9)


def test_transient_radiation_reaches_expected_equilibrium_direction() -> None:
    mesh, nodes = _build_slab()
    material = ThermalMaterial(thermal_conductivity=50.0, density=7850.0, specific_heat=460.0)
    right_face = ThermalSurface(1, _RIGHT_FACE_INDEX)

    analysis = TransientThermalAnalysis(
        mesh, {1: material}, time_step=5.0, num_steps=40, initial_temperature=300.0
    )
    for node in nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(PrescribedTemperature(node.id, 500.0))
    analysis.add_radiation(
        RadiationBoundaryCondition(
            surface=right_face, emissivity=0.8, surrounding_temperature=300.0
        )
    )
    result = analysis.solve()

    history = result.node_temperature_history(2)
    # Monotonic heating from the initial 300K toward (but below) the 500K hot face.
    assert np.all(np.diff(history) >= -1e-9)
    assert 300.0 < history[-1] < 500.0
