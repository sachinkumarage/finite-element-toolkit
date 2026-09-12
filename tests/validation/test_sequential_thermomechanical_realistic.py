"""Validation: the realistic sequential thermoelastic workflow (spec sections 14-16).

Improves on Version 20's sequential thermomechanical demonstration by
driving the thermal side with genuinely realistic Version 21 boundary
conditions -- one face held at a furnace-wall temperature, the opposite
face losing heat by both convection and radiation to ambient -- rather
than a hand-picked uniform or simple two-Dirichlet-face temperature. The
full workflow exercised here is exactly spec section 16's:

.. code-block:: text

    1. Solve the thermal problem (convection + radiation).
    2. Obtain nodal temperatures.
    3. Interpolate temperature at mechanical integration points
       (the existing Version 19 element-average convention).
    4. Evaluate temperature-dependent material properties (E(T), alpha(T)).
    5. Calculate thermal strain.
    6. Calculate mechanical response.
    7. Solve the mechanical problem.

No new coupling code is introduced -- Version 19's
:class:`~femtoolkit.analysis.temperature_field.TemperatureField` and
:func:`~femtoolkit.analysis.temperature_field.thermoelastic_materials_for_mesh`,
and Version 19's ``ThermoelasticMaterial3D`` (which already supports
``E(T)``/``v(T)``/``alpha(T)``), are reused completely unmodified. What
is new here is the thermal *input* to that unchanged pipeline: a
solved, physically-derived equilibrium temperature (from convection and
radiation) rather than an assumed one.
"""

import numpy as np
import pytest

from femtoolkit.analysis import BoundaryCondition, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.analysis.temperature_field import thermoelastic_materials_for_mesh
from femtoolkit.materials import LinearElastic3D, ThermoelasticMaterial3D
from femtoolkit.materials.thermal_properties import TemperatureDependentProperty
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.thermal import PrescribedTemperature, SteadyStateThermalAnalysis, ThermalMaterial
from femtoolkit.thermal.thermal_boundary_conditions import (
    ConvectionBoundaryCondition,
    RadiationBoundaryCondition,
)
from femtoolkit.thermal.thermal_surfaces import ThermalSurface

X = TranslationDOF.X
Y = TranslationDOF.Y
Z = TranslationDOF.Z

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]
_RIGHT_FACE_INDEX = 4
_REFERENCE_TEMPERATURE = 293.15


def _build_block() -> tuple[Mesh, Hex8Element3D]:
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)
    return mesh, hexa


def test_convection_and_radiation_driven_thermal_expansion_matches_closed_form() -> None:
    mesh, hexa = _build_block()

    thermal_material = ThermalMaterial(
        thermal_conductivity=50.0, density=7850.0, specific_heat=460.0
    )
    thermal_analysis = SteadyStateThermalAnalysis(mesh, {hexa.id: thermal_material})
    for node in hexa.nodes:
        if node.x == 0.0:
            thermal_analysis.add_boundary_condition(PrescribedTemperature(node.id, 600.0))
    right_face = ThermalSurface(hexa.id, _RIGHT_FACE_INDEX)
    thermal_analysis.add_convection(
        ConvectionBoundaryCondition(
            surface=right_face, convection_coefficient=15.0, ambient_temperature=300.0
        )
    )
    thermal_analysis.add_radiation(
        RadiationBoundaryCondition(
            surface=right_face, emissivity=0.7, surrounding_temperature=300.0
        )
    )
    thermal_result = thermal_analysis.solve()
    temperature_field = thermal_result.to_temperature_field()

    # Temperature-dependent mechanical properties (spec section 15): E(T) and alpha(T).
    youngs_modulus_table = TemperatureDependentProperty(
        temperatures=(293.15, 900.0), values=(200e9, 150e9)
    )
    thermal_expansion_table = TemperatureDependentProperty(
        temperatures=(293.15, 900.0), values=(12e-6, 16e-6)
    )
    base_material = ThermoelasticMaterial3D(
        youngs_modulus=youngs_modulus_table,
        poisson_ratio=0.3,
        thermal_expansion_coefficient=thermal_expansion_table,
        reference_temperature=_REFERENCE_TEMPERATURE,
        density=7850.0,
    )
    materials = thermoelastic_materials_for_mesh(mesh, base_material, temperature_field)
    element_temperature = materials[hexa.id].temperature

    # The element's temperature must lie strictly between the hot face and the
    # convection/radiation-cooled ambient -- a genuine, physically-derived value.
    assert 300.0 < element_temperature < 600.0

    settings = NonlinearSolverSettings(load_steps=1, tolerance=1e-6, max_iterations=15)
    mechanical_analysis = NonlinearAnalysis(mesh, materials, settings, geometric_nonlinearity=False)
    mechanical_analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(1, Z, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(2, Y, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(2, Z, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(4, Z, 0.0))
    result = mechanical_analysis.solve()
    assert result.converged

    alpha_at_element_temperature = base_material.thermal_expansion_coefficient_at(
        element_temperature
    )
    delta_temperature = element_temperature - _REFERENCE_TEMPERATURE
    expected_ux_node2 = alpha_at_element_temperature * delta_temperature * 1.0
    assert result.displacement(2, X) == pytest.approx(expected_ux_node2, rel=1e-9)

    for gauss_point in range(8):
        state = result.element_state(hexa.id, gauss_point=gauss_point)
        assert np.abs(state.stress).max() < 1.0e-3  # free expansion: ~0 Pa, FP noise only


def test_temperature_dependent_expansion_differs_from_constant_property_result() -> None:
    """Using T-dependent alpha must give a different displacement than freezing it at
    its reference-temperature value -- otherwise the temperature dependence has no effect."""
    mesh, hexa = _build_block()

    thermal_material = ThermalMaterial(
        thermal_conductivity=50.0, density=7850.0, specific_heat=460.0
    )
    thermal_analysis = SteadyStateThermalAnalysis(mesh, {hexa.id: thermal_material})
    for node in hexa.nodes:
        if node.x == 0.0:
            thermal_analysis.add_boundary_condition(PrescribedTemperature(node.id, 700.0))
    right_face = ThermalSurface(hexa.id, _RIGHT_FACE_INDEX)
    thermal_analysis.add_convection(
        ConvectionBoundaryCondition(
            surface=right_face, convection_coefficient=20.0, ambient_temperature=300.0
        )
    )
    thermal_result = thermal_analysis.solve()
    temperature_field = thermal_result.to_temperature_field()

    thermal_expansion_table = TemperatureDependentProperty(
        temperatures=(293.15, 900.0), values=(12e-6, 20e-6)
    )
    variable_material = ThermoelasticMaterial3D(
        youngs_modulus=200e9,
        poisson_ratio=0.3,
        thermal_expansion_coefficient=thermal_expansion_table,
        reference_temperature=_REFERENCE_TEMPERATURE,
        density=7850.0,
    )
    constant_material = ThermoelasticMaterial3D(
        youngs_modulus=200e9,
        poisson_ratio=0.3,
        thermal_expansion_coefficient=12e-6,
        reference_temperature=_REFERENCE_TEMPERATURE,
        density=7850.0,
    )

    settings = NonlinearSolverSettings(load_steps=1, tolerance=1e-6, max_iterations=15)

    def _solve_expansion(base_material: ThermoelasticMaterial3D) -> float:
        materials = thermoelastic_materials_for_mesh(mesh, base_material, temperature_field)
        analysis = NonlinearAnalysis(mesh, materials, settings, geometric_nonlinearity=False)
        analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
        analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
        analysis.add_boundary_condition(BoundaryCondition(1, Z, 0.0))
        analysis.add_boundary_condition(BoundaryCondition(2, Y, 0.0))
        analysis.add_boundary_condition(BoundaryCondition(2, Z, 0.0))
        analysis.add_boundary_condition(BoundaryCondition(4, Z, 0.0))
        result = analysis.solve()
        return result.displacement(2, X)

    variable_ux = _solve_expansion(variable_material)
    constant_ux = _solve_expansion(constant_material)
    assert variable_ux != pytest.approx(constant_ux)
    assert variable_ux > constant_ux  # alpha increases with temperature here
