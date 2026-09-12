"""Example: temperature-dependent mechanical properties, Version 21.

Version 19's ``ThermoelasticMaterial3D`` already supports temperature-
dependent Young's modulus and thermal expansion coefficient
(``E(T)``, ``alpha(T)``) -- this script demonstrates the effect using a
realistic, convection/radiation-derived equilibrium temperature (rather
than a hand-picked one), and compares against freezing both properties
at their room-temperature values to show the temperature dependence
genuinely changes the mechanical result.
"""

from femtoolkit.analysis import BoundaryCondition, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.analysis.temperature_field import TemperatureField, thermoelastic_materials_for_mesh
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

REFERENCE_TEMPERATURE = 293.15  # K
HOT_FACE_TEMPERATURE = 700.0  # K
AMBIENT_TEMPERATURE = 300.0  # K
CONVECTION_COEFFICIENT = 20.0  # W/(m^2*K)
EMISSIVITY = 0.75
RIGHT_FACE_INDEX = 4  # xi = +1, the x = 1 face -- see HEX8_FACES

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


def _solve_thermal(mesh: Mesh, hexa: Hex8Element3D) -> TemperatureField:
    thermal_material = ThermalMaterial(
        thermal_conductivity=25.0, density=7850.0, specific_heat=460.0
    )
    analysis = SteadyStateThermalAnalysis(mesh, {hexa.id: thermal_material})
    for node in hexa.nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(PrescribedTemperature(node.id, HOT_FACE_TEMPERATURE))
    right_face = ThermalSurface(hexa.id, RIGHT_FACE_INDEX)
    analysis.add_convection(
        ConvectionBoundaryCondition(
            surface=right_face,
            convection_coefficient=CONVECTION_COEFFICIENT,
            ambient_temperature=AMBIENT_TEMPERATURE,
        )
    )
    analysis.add_radiation(
        RadiationBoundaryCondition(
            surface=right_face, emissivity=EMISSIVITY, surrounding_temperature=AMBIENT_TEMPERATURE
        )
    )
    return analysis.solve().to_temperature_field()


def _solve_expansion(
    mesh: Mesh, base_material: ThermoelasticMaterial3D, field: TemperatureField
) -> float:
    materials = thermoelastic_materials_for_mesh(mesh, base_material, field)
    settings = NonlinearSolverSettings(load_steps=1, tolerance=1e-6, max_iterations=15)
    analysis = NonlinearAnalysis(mesh, materials, settings, geometric_nonlinearity=False)
    analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Z, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(2, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(2, Z, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, Z, 0.0))
    result = analysis.solve()
    return result.displacement(2, X)


def main() -> None:
    """Compare thermal expansion under constant vs. temperature-dependent properties."""
    mesh, hexa = _build_block()
    temperature_field = _solve_thermal(mesh, hexa)

    youngs_modulus_table = TemperatureDependentProperty(
        temperatures=(293.15, 900.0), values=(200e9, 140e9)
    )
    thermal_expansion_table = TemperatureDependentProperty(
        temperatures=(293.15, 900.0), values=(12e-6, 18e-6)
    )
    variable_material = ThermoelasticMaterial3D(
        youngs_modulus=youngs_modulus_table,
        poisson_ratio=0.3,
        thermal_expansion_coefficient=thermal_expansion_table,
        reference_temperature=REFERENCE_TEMPERATURE,
        density=7850.0,
    )
    constant_material = ThermoelasticMaterial3D(
        youngs_modulus=200e9,
        poisson_ratio=0.3,
        thermal_expansion_coefficient=12e-6,
        reference_temperature=REFERENCE_TEMPERATURE,
        density=7850.0,
    )

    variable_ux = _solve_expansion(mesh, variable_material, temperature_field)
    constant_ux = _solve_expansion(mesh, constant_material, temperature_field)

    element_temperature = thermoelastic_materials_for_mesh(
        mesh, variable_material, temperature_field
    )[hexa.id].temperature

    print("Finite Element Toolkit")
    print("Version 21 -- Temperature-Dependent Mechanical Properties")
    print("=" * 58)

    print(
        f"\nSolved (convection + radiation) element-average temperature: "
        f"{element_temperature:.2f} K"
    )
    print(f"E(T) table: {youngs_modulus_table.values} Pa at {youngs_modulus_table.temperatures} K")
    print(
        f"alpha(T) table: {thermal_expansion_table.values} 1/K at "
        f"{thermal_expansion_table.temperatures} K"
    )

    print("\nFree-expansion displacement at node 2:")
    print(f"    Constant E, alpha (room-temperature values):  {constant_ux:.6e} m")
    print(f"    Temperature-dependent E(T), alpha(T):         {variable_ux:.6e} m")

    print(
        "\nAt this element's solved temperature, alpha(T) is larger than its "
        "room-temperature value, so the temperature-dependent case expands "
        "more -- the mechanical response genuinely depends on the actual "
        "operating temperature, not just a fixed material constant."
    )


if __name__ == "__main__":
    main()
