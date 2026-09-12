"""Example: the full sequential thermomechanical workflow, Version 21.

Demonstrates every step of spec section 16's realistic sequential
thermoelastic workflow, driven by genuinely realistic Version 21
boundary conditions:

.. code-block:: text

    1. Solve the thermal problem (a hot face, and convection + radiation
       cooling the opposite face -- not a hand-picked temperature).
    2. Obtain nodal temperatures.
    3. Interpolate temperature at mechanical integration points
       (Version 19's element-average convention).
    4. Evaluate temperature-dependent material properties, E(T) and alpha(T).
    5. Calculate thermal strain.
    6. Calculate mechanical response.
    7. Solve the mechanical problem.

No new coupling code is needed: Version 19's ``TemperatureField`` and
``thermoelastic_materials_for_mesh`` connect Version 21's thermal solver
straight into the unmodified mechanical ``NonlinearAnalysis``.
"""

import numpy as np

from femtoolkit.analysis import BoundaryCondition, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.analysis.temperature_field import TemperatureField, thermoelastic_materials_for_mesh
from femtoolkit.materials import LinearElastic3D, ThermoelasticMaterial3D
from femtoolkit.materials.thermal_properties import TemperatureDependentProperty
from femtoolkit.materials.thermoelastic import ThermoelasticMaterialAtTemperature
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.results.nonlinear_result import NonlinearAnalysisResult
from femtoolkit.thermal import PrescribedTemperature, SteadyStateThermalAnalysis, ThermalMaterial
from femtoolkit.thermal.thermal_boundary_conditions import (
    ConvectionBoundaryCondition,
    RadiationBoundaryCondition,
)
from femtoolkit.thermal.thermal_surfaces import ThermalSurface

X = TranslationDOF.X
Y = TranslationDOF.Y
Z = TranslationDOF.Z

REFERENCE_TEMPERATURE = 293.15  # K
HOT_FACE_TEMPERATURE = 600.0  # K
AMBIENT_TEMPERATURE = 300.0  # K
CONVECTION_COEFFICIENT = 15.0  # W/(m^2*K)
EMISSIVITY = 0.7
RIGHT_FACE_INDEX = 4  # xi = +1, the x = 1 face -- see HEX8_FACES

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def main() -> None:
    """Run every step of the sequential thermomechanical workflow and report the result."""
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    # Step 1: thermal analysis, with realistic convection + radiation cooling.
    thermal_material = ThermalMaterial(
        thermal_conductivity=25.0, density=7850.0, specific_heat=460.0
    )
    thermal_analysis = SteadyStateThermalAnalysis(mesh, {hexa.id: thermal_material})
    for node in nodes:
        if node.x == 0.0:
            thermal_analysis.add_boundary_condition(
                PrescribedTemperature(node.id, HOT_FACE_TEMPERATURE)
            )
    right_face = ThermalSurface(hexa.id, RIGHT_FACE_INDEX)
    thermal_analysis.add_convection(
        ConvectionBoundaryCondition(
            surface=right_face,
            convection_coefficient=CONVECTION_COEFFICIENT,
            ambient_temperature=AMBIENT_TEMPERATURE,
        )
    )
    thermal_analysis.add_radiation(
        RadiationBoundaryCondition(
            surface=right_face, emissivity=EMISSIVITY, surrounding_temperature=AMBIENT_TEMPERATURE
        )
    )
    thermal_result = thermal_analysis.solve()

    # Step 2: nodal temperatures, as a Version 19 TemperatureField.
    temperature_field = thermal_result.to_temperature_field()

    # Step 3 + 4: temperature-dependent material properties, interpolated per element.
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
        reference_temperature=REFERENCE_TEMPERATURE,
        density=7850.0,
    )
    materials = thermoelastic_materials_for_mesh(mesh, base_material, temperature_field)

    # Steps 5-7: thermal strain, mechanical response, mechanical solve.
    settings = NonlinearSolverSettings(load_steps=1, tolerance=1e-6, max_iterations=15)
    mechanical_analysis = NonlinearAnalysis(mesh, materials, settings, geometric_nonlinearity=False)
    mechanical_analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(1, Z, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(2, Y, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(2, Z, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(4, Z, 0.0))
    result = mechanical_analysis.solve()

    print_summary(mesh, nodes, temperature_field, materials, hexa.id, result)


def print_summary(
    mesh: Mesh,
    nodes: tuple[Node, ...],
    temperature_field: TemperatureField,
    materials: dict[int, ThermoelasticMaterialAtTemperature],
    element_id: int,
    result: NonlinearAnalysisResult,
) -> None:
    """Print every stage of the sequential workflow."""
    print("Finite Element Toolkit")
    print("Version 21 -- Sequential Thermomechanical Workflow")
    print("=" * 52)

    print(f"\nStep 1: thermal analysis (hot face at {HOT_FACE_TEMPERATURE} K, opposite face")
    print(f"        convecting + radiating to {AMBIENT_TEMPERATURE} K ambient/surroundings)")

    print("\nStep 2: solved temperature field, per node:")
    for node in nodes:
        print(f"    Node {node.id}: T={temperature_field.node_temperature(node.id):.4f} K")

    element_temperature = materials[element_id].temperature
    print("\nStep 3-4: element-average temperature and evaluated properties:")
    print(f"    T = {element_temperature:.4f} K")
    base_material = materials[element_id].base_material
    print(f"    E(T) = {base_material.youngs_modulus_at(element_temperature):.4e} Pa")
    alpha = base_material.thermal_expansion_coefficient_at(element_temperature)
    print(f"    alpha(T) = {alpha:.6e} 1/K")

    print(f"\nStep 5-7: mechanical solve. Converged: {result.converged}")
    print("Nodal displacements:")
    for node in mesh.nodes:
        ux, uy, uz = result.node_displacement(node.id)
        print(f"    Node {node.id}: ux={ux:.4e}, uy={uy:.4e}, uz={uz:.4e} m")

    delta_temperature = element_temperature - REFERENCE_TEMPERATURE
    expected_ux = alpha * delta_temperature
    print(f"\nExpected free-expansion displacement, alpha(T)*dT: {expected_ux:.6e} m")

    print("\nMechanical stress (should be ~0 Pa -- free expansion develops no stress):")
    for gauss_point in range(8):
        state = result.element_state(element_id, gauss_point=gauss_point)
        print(f"    Point {gauss_point}: max|stress|={np.abs(state.stress).max():.3e} Pa")


if __name__ == "__main__":
    main()
