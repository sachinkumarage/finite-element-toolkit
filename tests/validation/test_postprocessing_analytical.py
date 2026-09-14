"""Validation: post-processing calculations against simple analytical fields (spec section 22).

For a linear temperature field ``T(x) = T0 + a*x``, the exact analytical
gradient and heat flux are:

.. code-block:: text

    grad(T) = a
    q = -k * a

This file solves that exact field through the real solver (not a
hand-built result), then verifies the *post-processing* layer --
:mod:`femtoolkit.postprocessing.adapters`,
:mod:`femtoolkit.postprocessing.field_calculator`, and
:mod:`femtoolkit.postprocessing.post_processor` together -- reports
exactly these values, end to end. It also verifies deformed-geometry
construction against a known displacement and verifies the equivalent
(von Mises) stress/strain measures against known, simple stress/strain
states (a uniaxial state, where the equivalent measure must equal the
axial value exactly).
"""

import numpy as np
import pytest

from femtoolkit.analysis import BoundaryCondition, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.materials import LinearElastic3D, ThermoelasticMaterial3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.postprocessing.adapters import from_nonlinear, from_thermal_steady_state
from femtoolkit.postprocessing.field_calculator import (
    deformed_coordinates,
    equivalent_strain,
    equivalent_stress,
    with_derived_fields,
)
from femtoolkit.postprocessing.post_processor import PostProcessor
from femtoolkit.postprocessing.result_model import MeshTopology
from femtoolkit.thermal import PrescribedTemperature, SteadyStateThermalAnalysis, ThermalMaterial

X = TranslationDOF.X
Y = TranslationDOF.Y
Z = TranslationDOF.Z

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def _build_block() -> tuple[Mesh, Hex8Element3D]:
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)
    return mesh, hexa


@pytest.mark.parametrize(("t0", "a", "conductivity"), [(300.0, 50.0, 20.0), (0.0, -10.0, 5.0)])
def test_temperature_gradient_and_heat_flux_match_linear_field(
    t0: float, a: float, conductivity: float
) -> None:
    """T(x) = T0 + a*x -> grad(T) = a, q = -k*a, verified through the full
    post-processing pipeline."""
    mesh, hexa = _build_block()
    material = ThermalMaterial(
        thermal_conductivity=conductivity, density=7850.0, specific_heat=460.0
    )
    analysis = SteadyStateThermalAnalysis(mesh, {hexa.id: material})
    for node in hexa.nodes:
        analysis.add_boundary_condition(PrescribedTemperature(node.id, t0 + a * node.x))
    result = analysis.solve()

    simulation = with_derived_fields(from_thermal_steady_state(result))
    processor = PostProcessor(simulation)

    gradient = simulation.final_step.element_value("temperature_gradient", hexa.id)
    assert gradient == pytest.approx([a, 0.0, 0.0], abs=1e-9)

    flux = simulation.final_step.element_value("heat_flux", hexa.id)
    assert flux == pytest.approx([-conductivity * a, 0.0, 0.0], abs=1e-6)

    gradient_magnitude = simulation.final_step.element_value(
        "temperature_gradient_magnitude", hexa.id
    )
    assert gradient_magnitude == pytest.approx(abs(a), abs=1e-9)

    assert processor.minimum("temperature") == pytest.approx(min(t0, t0 + a), abs=1e-9)
    assert processor.maximum("temperature") == pytest.approx(max(t0, t0 + a), abs=1e-9)
    assert processor.mean("temperature") == pytest.approx(t0 + a / 2.0, abs=1e-9)


def test_deformed_coordinates_match_known_displacement_exactly() -> None:
    mesh, _hexa = _build_block()
    topology = MeshTopology.from_mesh(mesh)
    displacement = {1: np.array([0.02, -0.01, 0.005])}
    deformed = deformed_coordinates(topology, displacement, scale=1.0)

    original = topology.node_coordinates[1]
    expected = tuple(np.array(original) + displacement[1])
    assert deformed[1] == pytest.approx(expected)


def test_deformed_coordinates_zero_displacement_reproduces_original_geometry() -> None:
    mesh, hexa = _build_block()
    topology = MeshTopology.from_mesh(mesh)
    displacement = {node.id: np.zeros(3) for node in hexa.nodes}
    deformed = deformed_coordinates(topology, displacement, scale=1.0)
    for node in hexa.nodes:
        assert deformed[node.id] == topology.node_coordinates[node.id]


@pytest.mark.parametrize("visualization_scale", [1.0, 10.0, 1000.0])
def test_deformed_coordinates_scale_factor_never_alters_the_underlying_displacement(
    visualization_scale: float,
) -> None:
    """The scale factor is purely cosmetic (spec section 11): the true displacement,
    queryable from the result itself, is unaffected regardless of what scale a plot uses."""
    mesh, _hexa = _build_block()
    topology = MeshTopology.from_mesh(mesh)
    true_displacement = np.array([0.001, 0.0, 0.0])
    displacement_field = {1: true_displacement}

    deformed_coordinates(topology, displacement_field, scale=visualization_scale)
    # The input dictionary itself -- the "true" physical result -- must be untouched.
    assert displacement_field[1] == pytest.approx(true_displacement)


def test_equivalent_stress_uniaxial_state_equals_axial_stress() -> None:
    sigma_axial = 150e6
    stress_3d = np.array([sigma_axial, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert equivalent_stress(stress_3d) == pytest.approx(sigma_axial)


def test_equivalent_strain_uniaxial_incompressible_state_equals_axial_strain() -> None:
    eps_axial = 0.015
    strain_3d = np.array([eps_axial, -0.5 * eps_axial, -0.5 * eps_axial, 0.0, 0.0, 0.0])
    assert equivalent_strain(strain_3d) == pytest.approx(eps_axial)


def test_free_thermal_expansion_equivalent_stress_is_near_zero_through_full_pipeline() -> None:
    """A known, closed-form mechanical state: free thermal expansion develops zero stress.
    Verifies the post-processing pipeline reports this correctly end to end, not just the
    raw material state (spec section 9's mandatory von Mises verification)."""
    mesh, hexa = _build_block()
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

    simulation = with_derived_fields(from_nonlinear(result, mesh))
    processor = PostProcessor(simulation)
    assert processor.maximum("von_mises_stress", kind="element") < 1.0
