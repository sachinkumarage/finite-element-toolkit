"""Validation: the sequential thermal -> mechanical workflow (spec section 15).

Thermal analysis -> Temperature field -> Thermal expansion -> Mechanical
analysis, with zero changes to Version 19's thermomechanical machinery:
a :class:`~femtoolkit.thermal.thermal_analysis.SteadyStateThermalAnalysis`
is solved on a HEX8/TET4 block, its
:meth:`~femtoolkit.thermal.thermal_result.SteadyStateThermalResult.to_temperature_field`
becomes a Version 19
:class:`~femtoolkit.analysis.temperature_field.TemperatureField`, which
feeds :func:`~femtoolkit.analysis.temperature_field.thermoelastic_materials_for_mesh`
and then :class:`~femtoolkit.analysis.nonlinear_analysis.NonlinearAnalysis`
exactly as ``tests/validation/test_thermoelastic_free_expansion.py`` does
with a hand-picked temperature -- here the temperature instead comes from
an actual heat-conduction solve.
"""

import numpy as np
import pytest

from femtoolkit.analysis import BoundaryCondition, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.analysis.temperature_field import thermoelastic_materials_for_mesh
from femtoolkit.materials import LinearElastic3D, ThermoelasticMaterial3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node, Tet4Element3D
from femtoolkit.thermal import PrescribedTemperature, SteadyStateThermalAnalysis, ThermalMaterial

X = TranslationDOF.X
Y = TranslationDOF.Y
Z = TranslationDOF.Z

_YOUNGS_MODULUS = 200e9
_POISSON_RATIO = 0.3
_ALPHA = 12e-6
_T_REF = 293.15
_CONDUCTIVITY = 50.0
_T_HOT = 393.15
_T_COLD = 293.15


def _thermoelastic_base_material() -> ThermoelasticMaterial3D:
    return ThermoelasticMaterial3D(
        youngs_modulus=_YOUNGS_MODULUS,
        poisson_ratio=_POISSON_RATIO,
        thermal_expansion_coefficient=_ALPHA,
        reference_temperature=_T_REF,
        density=7850.0,
    )


def test_hex8_sequential_thermal_to_mechanical_workflow() -> None:
    coords = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(coords))
    placeholder = LinearElastic3D(
        youngs_modulus=_YOUNGS_MODULUS, poisson_ratio=_POISSON_RATIO, density=7850.0
    )
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)

    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    # Step 1: thermal analysis. Faces at x=0 and x=1 held at fixed temperatures.
    thermal_material = ThermalMaterial(
        thermal_conductivity=_CONDUCTIVITY, density=7850.0, specific_heat=460.0
    )
    thermal_analysis = SteadyStateThermalAnalysis(mesh, {hexa.id: thermal_material})
    for node in nodes:
        thermal_analysis.add_boundary_condition(
            PrescribedTemperature(node.id, _T_HOT if node.x == 0.0 else _T_COLD)
        )
    thermal_result = thermal_analysis.solve()

    # Step 2: temperature field, bridging into Version 19.
    temperature_field = thermal_result.to_temperature_field()
    for node in nodes:
        expected = _T_HOT if node.x == 0.0 else _T_COLD
        assert temperature_field.node_temperature(node.id) == pytest.approx(expected)

    # Step 3: thermal expansion materials, one per element, at that element's temperature.
    base_material = _thermoelastic_base_material()
    materials = thermoelastic_materials_for_mesh(mesh, base_material, temperature_field)
    expected_element_temperature = (_T_HOT + _T_COLD) / 2.0
    assert materials[hexa.id].temperature == pytest.approx(expected_element_temperature)

    # Step 4: mechanical analysis under that thermal expansion, minimally constrained
    # (rigid-body modes only) so the block is free to expand.
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

    delta_temperature = expected_element_temperature - _T_REF
    expected_ux_node2 = _ALPHA * delta_temperature * 1.0
    assert result.displacement(2, X) == pytest.approx(expected_ux_node2, rel=1e-6)

    for gauss_point in range(8):
        state = result.element_state(1, gauss_point=gauss_point)
        assert np.abs(state.stress).max() < 1.0  # free expansion: ~0 Pa, floating-point noise only


def test_tet4_sequential_thermal_to_mechanical_workflow() -> None:
    nodes = (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=1.0, y=0.0, z=0.0),
        Node(id=3, x=0.0, y=1.0, z=0.0),
        Node(id=4, x=0.0, y=0.0, z=1.0),
    )
    placeholder = LinearElastic3D(
        youngs_modulus=_YOUNGS_MODULUS, poisson_ratio=_POISSON_RATIO, density=7850.0
    )
    tet = Tet4Element3D(id=1, nodes=nodes, material=placeholder)

    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(tet)

    # A uniform temperature rise (all four nodes held at the same value) -- the
    # thermal solve degenerates to "every node is at the prescribed value," the
    # simplest possible sequential case, but still exercised through the real solver.
    thermal_material = ThermalMaterial(
        thermal_conductivity=_CONDUCTIVITY, density=7850.0, specific_heat=460.0
    )
    thermal_analysis = SteadyStateThermalAnalysis(mesh, {tet.id: thermal_material})
    for node in nodes:
        thermal_analysis.add_boundary_condition(PrescribedTemperature(node.id, _T_HOT))
    thermal_result = thermal_analysis.solve()
    temperature_field = thermal_result.to_temperature_field()

    base_material = _thermoelastic_base_material()
    materials = thermoelastic_materials_for_mesh(mesh, base_material, temperature_field)
    assert materials[tet.id].temperature == pytest.approx(_T_HOT)

    settings = NonlinearSolverSettings(load_steps=1, tolerance=1e-6, max_iterations=15)
    mechanical_analysis = NonlinearAnalysis(mesh, materials, settings, geometric_nonlinearity=False)
    mechanical_analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(1, Z, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(2, Y, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(2, Z, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(3, Z, 0.0))

    result = mechanical_analysis.solve()
    assert result.converged

    delta_temperature = _T_HOT - _T_REF
    expected_ux_node2 = _ALPHA * delta_temperature * 1.0
    assert result.displacement(2, X) == pytest.approx(expected_ux_node2, rel=1e-6)
    state = result.element_state(1)
    assert np.abs(state.stress).max() < 1.0
