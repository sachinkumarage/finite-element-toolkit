"""Tests for femtoolkit.postprocessing.adapters."""

import numpy as np
import pytest

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.dynamic_analysis import DynamicAnalysis
from femtoolkit.analysis.dynamic_loads import StepLoad, TimeDependentNodalLoad
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.analysis.static_linear import StaticLinearAnalysis
from femtoolkit.analysis.temperature_field import thermoelastic_materials_for_mesh
from femtoolkit.materials import (
    LinearElastic2D,
    LinearElastic3D,
    Material,
    ThermoelasticMaterial3D,
)
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.mesh.bar_element import BarElement
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.postprocessing.adapters import (
    from_dynamic,
    from_nonlinear,
    from_static_linear,
    from_thermal_steady_state,
    from_thermal_transient,
    merge_thermomechanical,
)
from femtoolkit.sections import CrossSection
from femtoolkit.thermal import PrescribedTemperature, SteadyStateThermalAnalysis, ThermalMaterial
from femtoolkit.thermal.thermal_analysis import TransientThermalAnalysis

_HEX8_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]

X = TranslationDOF.X
Y = TranslationDOF.Y
Z = TranslationDOF.Z


def _hex8_mesh() -> tuple[Mesh, Hex8Element3D]:
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_HEX8_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)
    return mesh, hexa


def test_from_thermal_steady_state() -> None:
    mesh, hexa = _hex8_mesh()
    material = ThermalMaterial(thermal_conductivity=50.0, density=7850.0, specific_heat=460.0)
    analysis = SteadyStateThermalAnalysis(mesh, {hexa.id: material})
    for node in hexa.nodes:
        analysis.add_boundary_condition(
            PrescribedTemperature(node.id, 400.0 if node.x == 0.0 else 300.0)
        )
    result = analysis.solve()

    simulation = from_thermal_steady_state(result)
    assert simulation.num_steps == 1
    assert simulation.final_step.nodal_value("temperature", 1) == pytest.approx(400.0)
    flux = simulation.final_step.element_value("heat_flux", hexa.id)
    assert flux[0] == pytest.approx(-50.0 * (300.0 - 400.0))
    assert set(simulation.topology.node_ids) == {node.id for node in hexa.nodes}


def test_from_thermal_transient() -> None:
    mesh, hexa = _hex8_mesh()
    material = ThermalMaterial(thermal_conductivity=50.0, density=7850.0, specific_heat=460.0)
    analysis = TransientThermalAnalysis(
        mesh, {hexa.id: material}, time_step=100.0, num_steps=3, initial_temperature=293.15
    )
    for node in hexa.nodes:
        analysis.add_boundary_condition(
            PrescribedTemperature(node.id, 400.0 if node.x == 0.0 else 300.0)
        )
    result = analysis.solve()

    simulation = from_thermal_transient(result)
    assert simulation.num_steps == 4  # initial condition + 3 steps
    assert simulation.times[0] == pytest.approx(0.0)
    history = simulation.nodal_history("temperature", 1)
    assert history[0] == pytest.approx(400.0)
    assert "temperature_gradient" in simulation.final_step.element_fields
    assert "heat_flux" in simulation.final_step.element_fields


def test_from_static_linear() -> None:
    material = Material(name="steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    n1 = Node(1, 0.0, 0.0, 0.0)
    n2 = Node(2, 1.0, 0.0, 0.0)
    bar = BarElement(id=1, nodes=(n1, n2), material=material, cross_section=CrossSection(area=0.01))
    mesh = Mesh()
    mesh.add_node(n1)
    mesh.add_node(n2)
    mesh.add_element(bar)

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    analysis.add_load(NodalLoad(2, X, 1000.0))
    result = analysis.solve()

    simulation = from_static_linear(result)
    assert simulation.num_steps == 1
    assert simulation.final_step.nodal_value("displacement", 2)[0] == pytest.approx(5e-7)
    assert simulation.final_step.element_value("stress", 1)[0] == pytest.approx(1e5)
    assert simulation.topology.element_connectivity[1] == (1, 2)


def test_from_nonlinear() -> None:
    mesh, hexa = _hex8_mesh()
    base_material = ThermoelasticMaterial3D(
        youngs_modulus=200e9,
        poisson_ratio=0.3,
        thermal_expansion_coefficient=12e-6,
        reference_temperature=293.15,
        density=7850.0,
    )
    material = base_material.at_temperature(393.15)
    settings = NonlinearSolverSettings(load_steps=1, tolerance=1e-6, max_iterations=15)
    analysis = NonlinearAnalysis(mesh, {hexa.id: material}, settings, geometric_nonlinearity=False)
    analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Z, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(2, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(2, Z, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, Z, 0.0))
    result = analysis.solve()

    simulation = from_nonlinear(result, mesh)
    assert simulation.num_steps == 1
    assert simulation.times[0] == pytest.approx(1.0)
    displacement = simulation.final_step.nodal_value("displacement", 2)
    assert displacement[0] == pytest.approx(12e-6 * 100.0, rel=1e-6)
    stress = simulation.final_step.element_value("stress", hexa.id)
    assert np.abs(stress).max() < 1.0
    assert "plastic_strain" in simulation.final_step.element_fields


def test_from_dynamic() -> None:
    material = LinearElastic2D(
        youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress", density=7850.0
    )
    n1 = Node(1, 0.0, 0.0, 0.0)
    n2 = Node(2, 1.0, 0.0, 0.0)
    n3 = Node(3, 0.0, 1.0, 0.0)
    cst = CSTElement2D(id=1, nodes=(n1, n2, n3), material=material, thickness=0.01)
    mesh = Mesh()
    mesh.add_node(n1)
    mesh.add_node(n2)
    mesh.add_node(n3)
    mesh.add_element(cst)

    analysis = DynamicAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(3, X, 0.0))
    analysis.add_time_dependent_load(TimeDependentNodalLoad(2, X, StepLoad(1e6)))
    result = analysis.solve(time_step=0.0001, total_time=0.0003)

    simulation = from_dynamic(result, mesh)
    assert simulation.num_steps == len(result.time)
    assert simulation.final_step.nodal_value("displacement", 2)[0] == pytest.approx(
        result.displacement(2, X)[-1]
    )
    assert "stress" in simulation.final_step.element_fields


def test_merge_thermomechanical_combines_temperature_and_mechanical_fields() -> None:
    mesh, hexa = _hex8_mesh()
    thermal_material = ThermalMaterial(
        thermal_conductivity=50.0, density=7850.0, specific_heat=460.0
    )
    thermal_analysis = SteadyStateThermalAnalysis(mesh, {hexa.id: thermal_material})
    for node in hexa.nodes:
        thermal_analysis.add_boundary_condition(PrescribedTemperature(node.id, 500.0))
    thermal_result = thermal_analysis.solve()
    temperature_field = thermal_result.to_temperature_field()

    base_material = ThermoelasticMaterial3D(
        youngs_modulus=200e9,
        poisson_ratio=0.3,
        thermal_expansion_coefficient=12e-6,
        reference_temperature=293.15,
        density=7850.0,
    )
    materials = thermoelastic_materials_for_mesh(mesh, base_material, temperature_field)
    settings = NonlinearSolverSettings(load_steps=1, tolerance=1e-6, max_iterations=15)
    mechanical_analysis = NonlinearAnalysis(mesh, materials, settings, geometric_nonlinearity=False)
    mechanical_analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(1, Z, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(2, Y, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(2, Z, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(4, Z, 0.0))
    mechanical_result = mechanical_analysis.solve()

    thermal_simulation = from_thermal_steady_state(thermal_result)
    mechanical_simulation = from_nonlinear(mechanical_result, mesh)
    merged = merge_thermomechanical(thermal_simulation, mechanical_simulation)

    assert merged.topology is mechanical_simulation.topology
    assert merged.final_step.nodal_value("temperature", 2) == pytest.approx(500.0)
    assert merged.final_step.nodal_value(
        "displacement", 2
    )[0] == pytest.approx(mechanical_simulation.final_step.nodal_value("displacement", 2)[0])
    assert "heat_flux" in merged.final_step.element_fields
