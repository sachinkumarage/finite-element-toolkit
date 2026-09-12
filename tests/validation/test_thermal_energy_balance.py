"""Validation: thermal energy balance (spec sections 20-21).

For a steady-state problem, ``Q_in ~= Q_out``: every watt supplied by
heat generation, prescribed flux, and net convective/radiative exchange
must be carried away through the prescribed-temperature boundaries (see
:func:`~femtoolkit.thermal.thermal_analysis.steady_state_energy_balance`).

For a transient problem, ``Q_in - Q_out ~= dU``: the net heat flowing in
over one Backward-Euler step must equal the change in the domain's
stored thermal energy over that step. Since every shape function set
satisfies the partition of unity (``sum_i(Ni) = 1``), summing the
capacity matrix against a vector of ones recovers exactly
``integral(rho*c*T) dV`` -- the domain's total stored thermal energy
relative to a zero-kelvin reference -- with no extra machinery needed.
This file reconstructs that identity directly from the same building
blocks (:func:`~femtoolkit.thermal.thermal_elements.conductivity_contribution`,
:func:`~femtoolkit.thermal.thermal_elements.capacity_contribution`, and
the private :func:`~femtoolkit.thermal.thermal_analysis._nonlinear_surface_terms`)
the transient solver itself uses, as an independent numerical check.
"""

import numpy as np
import pytest

from femtoolkit.analysis.assembly import assemble_global_mass, assemble_global_stiffness
from femtoolkit.materials import LinearElastic2D, LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.thermal import (
    HeatGeneration,
    PrescribedHeatFlux,
    PrescribedTemperature,
    SteadyStateThermalAnalysis,
    ThermalMaterial,
    TransientThermalAnalysis,
    heat_generation_to_thermal_loads,
)
from femtoolkit.thermal.thermal_analysis import (
    _nonlinear_surface_terms,
    steady_state_energy_balance,
)
from femtoolkit.thermal.thermal_boundary_conditions import (
    ConvectionBoundaryCondition,
    RadiationBoundaryCondition,
)
from femtoolkit.thermal.thermal_elements import capacity_contribution, conductivity_contribution
from femtoolkit.thermal.thermal_surfaces import ThermalSurface

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]
_RIGHT_FACE_INDEX = 4


def _build_slab() -> tuple[Mesh, Hex8Element3D]:
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)
    return mesh, hexa


def test_steady_state_balance_for_pure_conduction() -> None:
    mesh, hexa = _build_slab()
    material = ThermalMaterial(thermal_conductivity=50.0, density=7850.0, specific_heat=460.0)
    analysis = SteadyStateThermalAnalysis(mesh, {hexa.id: material})
    for node in hexa.nodes:
        temperature = 400.0 if node.x == 0.0 else 300.0
        analysis.add_boundary_condition(PrescribedTemperature(node.id, temperature))
    result = analysis.solve()
    q_in, q_out = steady_state_energy_balance(analysis, result)
    assert q_in == pytest.approx(q_out, abs=1e-6)


def test_steady_state_balance_for_heat_generation() -> None:
    mesh, hexa = _build_slab()
    material = ThermalMaterial(thermal_conductivity=50.0, density=7850.0, specific_heat=460.0)
    analysis = SteadyStateThermalAnalysis(mesh, {hexa.id: material})
    for node in hexa.nodes:
        analysis.add_boundary_condition(PrescribedTemperature(node.id, 300.0))
    generation = HeatGeneration(volumetric_rate=1.0e5)
    analysis.add_thermal_loads(heat_generation_to_thermal_loads(mesh, generation))
    result = analysis.solve()
    q_in, q_out = steady_state_energy_balance(analysis, result)
    assert q_in == pytest.approx(1.0e5, rel=1e-9)
    assert q_in == pytest.approx(q_out, rel=1e-9)


def test_steady_state_balance_for_convection_and_radiation() -> None:
    mesh, hexa = _build_slab()
    material = ThermalMaterial(thermal_conductivity=50.0, density=7850.0, specific_heat=460.0)
    right_face = ThermalSurface(hexa.id, _RIGHT_FACE_INDEX)
    analysis = SteadyStateThermalAnalysis(mesh, {hexa.id: material})
    for node in hexa.nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(PrescribedTemperature(node.id, 500.0))
    analysis.add_convection(
        ConvectionBoundaryCondition(
            surface=right_face, convection_coefficient=15.0, ambient_temperature=300.0
        )
    )
    analysis.add_radiation(
        RadiationBoundaryCondition(
            surface=right_face, emissivity=0.7, surrounding_temperature=300.0
        )
    )
    result = analysis.solve()
    q_in, q_out = steady_state_energy_balance(analysis, result)
    assert q_in == pytest.approx(q_out, rel=1e-8)


def test_steady_state_balance_for_cst_plate_with_prescribed_flux() -> None:
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
    material = ThermalMaterial(thermal_conductivity=50.0, density=7850.0, specific_heat=460.0)

    analysis = SteadyStateThermalAnalysis(mesh, {1: material, 2: material})
    analysis.add_boundary_condition(PrescribedTemperature(1, 300.0))
    analysis.add_boundary_condition(PrescribedTemperature(4, 300.0))
    analysis.add_heat_flux(PrescribedHeatFlux(2, 50.0))
    analysis.add_heat_flux(PrescribedHeatFlux(3, 50.0))
    result = analysis.solve()
    q_in, q_out = steady_state_energy_balance(analysis, result)
    assert q_in == pytest.approx(100.0, rel=1e-9)
    assert q_in == pytest.approx(q_out, rel=1e-9)


def test_transient_energy_balance_heat_entering_equals_heat_leaving_plus_stored() -> None:
    """Q_in - Q_out ~= dU across one Backward-Euler step (spec section 21)."""
    mesh, hexa = _build_slab()
    conductivity, coefficient, hot_temperature, ambient_temperature = 50.0, 10.0, 500.0, 300.0
    material = ThermalMaterial(
        thermal_conductivity=conductivity, density=7850.0, specific_heat=460.0
    )
    right_face = ThermalSurface(hexa.id, _RIGHT_FACE_INDEX)

    analysis = TransientThermalAnalysis(
        mesh, {hexa.id: material}, time_step=20.0, num_steps=10, initial_temperature=300.0
    )
    for node in hexa.nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(PrescribedTemperature(node.id, hot_temperature))
    analysis.add_convection(
        ConvectionBoundaryCondition(
            surface=right_face,
            convection_coefficient=coefficient,
            ambient_temperature=ambient_temperature,
        )
    )
    result = analysis.solve()

    dof_map = result.dof_map
    k_conduction = assemble_global_stiffness(
        dof_map, [conductivity_contribution(hexa, material, analysis.evaluation_temperature)]
    )
    c_t = assemble_global_mass(
        dof_map, [capacity_contribution(hexa, material, analysis.evaluation_temperature)]
    )

    for step in range(1, len(result.times)):
        temperature_old = result.temperature_history[step - 1]
        temperature_new = result.temperature_history[step]
        current_time = result.times[step]

        nonlinear_residual, _ = _nonlinear_surface_terms(
            mesh,
            analysis._convections,
            analysis._radiations,
            dof_map,
            temperature_new,
            current_time,
        )
        full_residual = (
            nonlinear_residual
            + (c_t @ temperature_old) / analysis.time_step
            - (c_t / analysis.time_step + k_conduction) @ temperature_new
        )
        constrained_indices = [
            dof_map.global_index(bc.node_id, 0) for bc in analysis._boundary_conditions
        ]

        q_supplied_rate = np.sum(nonlinear_residual)
        q_reaction_rate = -np.sum(full_residual[constrained_indices])
        predicted_delta_u = analysis.time_step * (q_supplied_rate + q_reaction_rate)
        actual_delta_u = np.sum(c_t @ (temperature_new - temperature_old))

        assert predicted_delta_u == pytest.approx(actual_delta_u, rel=1e-8)
