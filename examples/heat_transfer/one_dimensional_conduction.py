"""Example: 1D steady-state heat conduction along a bar, Version 20.

The toolkit's mandatory analytical benchmark (see
``tests/validation/test_thermal_steady_state_conduction.py``): a bar
with constant conductivity and no internal heat generation has
``d^2T/dx^2 = 0``, whose exact solution is the linear profile
``T(x) = T0 + (TL - T0) * x / L``. This script builds that bar with a
handful of 1D thermal conduction elements, solves ``K_T T = F_T``, and
compares the finite element temperatures and heat flux directly against
the closed form.
"""

from femtoolkit.materials import Material
from femtoolkit.mesh import Mesh, Node
from femtoolkit.mesh.bar_element import BarElement
from femtoolkit.sections import CrossSection
from femtoolkit.thermal import (
    PrescribedTemperature,
    SteadyStateThermalAnalysis,
    SteadyStateThermalResult,
    ThermalMaterial,
)

CONDUCTIVITY = 50.0  # W/(m*K), typical carbon steel
DENSITY = 7850.0  # kg/m^3
SPECIFIC_HEAT = 460.0  # J/(kg*K)
LENGTH = 2.0  # m
AREA = 0.01  # m^2
NUM_ELEMENTS = 5
T_HOT = 373.15  # K (100 C)
T_COLD = 293.15  # K (20 C)


def main() -> None:
    """Build, solve, and report a 1D conduction bar against its analytical solution."""
    mechanical_material = Material(
        name="steel", density=DENSITY, youngs_modulus=200e9, poissons_ratio=0.3
    )
    thermal_material = ThermalMaterial(
        thermal_conductivity=CONDUCTIVITY, density=DENSITY, specific_heat=SPECIFIC_HEAT
    )

    nodes = [
        Node(id=i + 1, x=LENGTH * i / NUM_ELEMENTS, y=0.0, z=0.0) for i in range(NUM_ELEMENTS + 1)
    ]
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)

    materials: dict[int, ThermalMaterial] = {}
    for i in range(NUM_ELEMENTS):
        element = BarElement(
            id=i + 1,
            nodes=(nodes[i], nodes[i + 1]),
            material=mechanical_material,
            cross_section=CrossSection(area=AREA),
        )
        mesh.add_element(element)
        materials[element.id] = thermal_material

    analysis = SteadyStateThermalAnalysis(mesh, materials)
    analysis.add_boundary_condition(PrescribedTemperature(nodes[0].id, T_HOT))
    analysis.add_boundary_condition(PrescribedTemperature(nodes[-1].id, T_COLD))

    result = analysis.solve()
    print_summary(nodes, materials, result)


def print_summary(
    nodes: list[Node], materials: dict[int, ThermalMaterial], result: SteadyStateThermalResult
) -> None:
    """Print nodal temperatures and heat flux against the closed-form linear solution."""
    print("Finite Element Toolkit")
    print("Version 20 -- 1D Steady-State Heat Conduction")
    print("=" * 50)

    print(f"\nConductivity: {CONDUCTIVITY} W/(m*K), Length: {LENGTH} m, Area: {AREA} m^2")
    print(f"Boundary temperatures: T(0) = {T_HOT} K, T(L) = {T_COLD} K")

    print("\nNodal temperatures (finite element vs. analytical T0 + (TL-T0)*x/L):")
    for node in nodes:
        analytical = T_HOT + (T_COLD - T_HOT) * node.x / LENGTH
        solved = result.node_temperature(node.id)
        print(f"    x={node.x:.3f} m: FE={solved:.4f} K, analytical={analytical:.4f} K")

    expected_flux = -CONDUCTIVITY * (T_COLD - T_HOT) / LENGTH
    print(f"\nExpected heat flux, q = -k*dT/dx: {expected_flux:.4f} W/m^2")
    print("Solved heat flux per element:")
    for element_id in materials:
        flux = result.element_heat_flux(element_id)
        print(f"    Element {element_id}: q_x = {flux[0]:.4f} W/m^2")

    print(
        "\nHeat flows from hot to cold (positive x-direction here), and every "
        "element carries the same flux -- steady state with no internal generation."
    )


if __name__ == "__main__":
    main()
