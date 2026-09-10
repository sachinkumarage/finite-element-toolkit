"""Example: sequential thermal -> mechanical analysis, Version 20.

Demonstrates the full sequential thermomechanical workflow this version
was built to connect to Version 19:

    Thermal analysis -> Temperature field -> Thermal expansion -> Mechanical analysis

A HEX8 block is first solved for its steady-state temperature
distribution (one face hot, the opposite face cold). The solved nodal
temperatures become a
:class:`~femtoolkit.analysis.temperature_field.TemperatureField` via
:meth:`~femtoolkit.thermal.thermal_result.SteadyStateThermalResult.to_temperature_field`,
which then drives Version 19's
:func:`~femtoolkit.analysis.temperature_field.thermoelastic_materials_for_mesh`
and :class:`~femtoolkit.analysis.nonlinear_analysis.NonlinearAnalysis`
completely unmodified: the block, minimally constrained against rigid
body motion, expands under its own thermal gradient.
"""

import numpy as np

from femtoolkit.analysis import BoundaryCondition, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.analysis.temperature_field import TemperatureField, thermoelastic_materials_for_mesh
from femtoolkit.materials import LinearElastic3D, ThermoelasticMaterial3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.results.nonlinear_result import NonlinearAnalysisResult
from femtoolkit.thermal import PrescribedTemperature, SteadyStateThermalAnalysis, ThermalMaterial

X = TranslationDOF.X
Y = TranslationDOF.Y
Z = TranslationDOF.Z

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
THERMAL_EXPANSION_COEFFICIENT = 12e-6  # 1/K
REFERENCE_TEMPERATURE = 293.15  # K
DENSITY = 7850.0  # kg/m^3
CONDUCTIVITY = 50.0  # W/(m*K)
SPECIFIC_HEAT = 460.0  # J/(kg*K)
T_HOT = 393.15  # K
T_COLD = 293.15  # K

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def main() -> None:
    """Run the sequential thermal-then-mechanical workflow on a HEX8 block."""
    placeholder_material = LinearElastic3D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, density=DENSITY
    )
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder_material)

    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    # Step 1: thermal analysis.
    thermal_material = ThermalMaterial(
        thermal_conductivity=CONDUCTIVITY, density=DENSITY, specific_heat=SPECIFIC_HEAT
    )
    thermal_analysis = SteadyStateThermalAnalysis(mesh, {hexa.id: thermal_material})
    for node in nodes:
        thermal_analysis.add_boundary_condition(
            PrescribedTemperature(node.id, T_HOT if node.x == 0.0 else T_COLD)
        )
    thermal_result = thermal_analysis.solve()

    # Step 2: temperature field.
    temperature_field = thermal_result.to_temperature_field()

    # Step 3: thermal expansion materials, one per element.
    base_material = ThermoelasticMaterial3D(
        youngs_modulus=YOUNGS_MODULUS,
        poisson_ratio=POISSON_RATIO,
        thermal_expansion_coefficient=THERMAL_EXPANSION_COEFFICIENT,
        reference_temperature=REFERENCE_TEMPERATURE,
        density=DENSITY,
    )
    materials = thermoelastic_materials_for_mesh(mesh, base_material, temperature_field)

    # Step 4: mechanical analysis, minimally constrained (rigid body modes only).
    settings = NonlinearSolverSettings(load_steps=1, tolerance=1e-6, max_iterations=15)
    mechanical_analysis = NonlinearAnalysis(mesh, materials, settings, geometric_nonlinearity=False)
    mechanical_analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(1, Z, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(2, Y, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(2, Z, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(4, Z, 0.0))
    result = mechanical_analysis.solve()

    print_summary(mesh, nodes, temperature_field, materials, result)


def print_summary(
    mesh: Mesh,
    nodes: tuple[Node, ...],
    temperature_field: TemperatureField,
    materials: dict[int, ThermoelasticMaterial3D],
    result: NonlinearAnalysisResult,
) -> None:
    """Print each stage of the sequential thermal-to-mechanical workflow."""
    print("Finite Element Toolkit")
    print("Version 20 -- Sequential Thermomechanical Workflow")
    print("=" * 55)

    print(f"\nStep 1: thermal analysis (x=0 face at {T_HOT} K, x=1 face at {T_COLD} K)")
    print("Step 2: solved temperature field, per node:")
    for node in nodes:
        print(f"    Node {node.id}: T={temperature_field.node_temperature(node.id):.4f} K")

    element_temperature = materials[1].temperature
    print(
        f"\nStep 3: element-average temperature used for thermal expansion: "
        f"{element_temperature:.4f} K"
    )

    print(f"\nStep 4: mechanical analysis. Converged: {result.converged}")
    print("Nodal displacements (free thermal expansion, minimally constrained):")
    for node in mesh.nodes:
        ux, uy, uz = result.node_displacement(node.id)
        print(f"    Node {node.id}: ux={ux:.3e}, uy={uy:.3e}, uz={uz:.3e} m")

    delta_temperature = element_temperature - REFERENCE_TEMPERATURE
    expected_ux = THERMAL_EXPANSION_COEFFICIENT * delta_temperature
    print(f"\nExpected free-expansion displacement, alpha*dT: {expected_ux:.6e} m")

    print("\nMechanical stress (should be ~0 Pa -- free expansion develops no stress):")
    for gauss_point in range(8):
        state = result.element_state(1, gauss_point=gauss_point)
        print(f"    Point {gauss_point}: max|stress|={np.abs(state.stress).max():.3e} Pa")

    print(
        "\nThe thermal solver's own solution -- not a hand-picked temperature -- "
        "drove the mechanical expansion, with no changes to Version 19's "
        "thermomechanical code."
    )


if __name__ == "__main__":
    main()
