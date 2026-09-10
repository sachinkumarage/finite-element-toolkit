"""Example: 1D conduction with uniform volumetric heat generation, Version 20.

For a bar held at the same temperature at both ends, with a uniform
volumetric heat source ``Q`` and no other loading,
``d/dx(k dT/dx) + Q = 0`` has the closed-form parabolic solution
``T(x) = T0 + (Q / (2k)) * x * (L - x)`` -- the temperature peaks at the
bar's midpoint, where the heat generated on either side must escape in
opposite directions. This mirrors
``tests/validation/test_thermal_solid_elements.py``'s heat-generation
benchmark, using a moderately fine mesh since (unlike the pure-conduction
case) the true solution is quadratic and linear elements only
approximate it.
"""

from femtoolkit.materials import Material
from femtoolkit.mesh import Mesh, Node
from femtoolkit.mesh.bar_element import BarElement
from femtoolkit.sections import CrossSection
from femtoolkit.thermal import (
    HeatGeneration,
    PrescribedTemperature,
    SteadyStateThermalAnalysis,
    SteadyStateThermalResult,
    ThermalMaterial,
    heat_generation_to_thermal_loads,
)

CONDUCTIVITY = 50.0  # W/(m*K)
DENSITY = 7850.0  # kg/m^3
SPECIFIC_HEAT = 460.0  # J/(kg*K)
LENGTH = 2.0  # m
AREA = 0.01  # m^2
NUM_ELEMENTS = 20
VOLUMETRIC_RATE = 1.0e5  # W/m^3
T_BOUNDARY = 293.15  # K, both ends


def main() -> None:
    """Build, solve, and report a heated bar against the parabolic closed form."""
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
    analysis.add_boundary_condition(PrescribedTemperature(nodes[0].id, T_BOUNDARY))
    analysis.add_boundary_condition(PrescribedTemperature(nodes[-1].id, T_BOUNDARY))
    heat_generation = HeatGeneration(volumetric_rate=VOLUMETRIC_RATE)
    analysis.add_thermal_loads(heat_generation_to_thermal_loads(mesh, heat_generation))

    result = analysis.solve()
    print_summary(nodes, result)


def print_summary(nodes: list[Node], result: SteadyStateThermalResult) -> None:
    """Print nodal temperatures against the closed-form parabolic solution."""
    print("Finite Element Toolkit")
    print("Version 20 -- Volumetric Heat Generation")
    print("=" * 50)

    print(f"\nUniform volumetric heat generation: Q = {VOLUMETRIC_RATE:.3e} W/m^3")
    print(f"Both ends fixed at T = {T_BOUNDARY} K, length = {LENGTH} m")

    max_temperature_rise = VOLUMETRIC_RATE * LENGTH**2 / (8.0 * CONDUCTIVITY)
    print(f"Expected peak temperature rise at midpoint: {max_temperature_rise:.4f} K")

    print("\nNodal temperatures (finite element vs. analytical T0 + Q/(2k)*x*(L-x)):")
    for node in nodes:
        analytical = T_BOUNDARY + (VOLUMETRIC_RATE / (2.0 * CONDUCTIVITY)) * node.x * (
            LENGTH - node.x
        )
        solved = result.node_temperature(node.id)
        print(f"    x={node.x:.3f} m: FE={solved:.4f} K, analytical={analytical:.4f} K")

    print(
        "\nThe temperature peaks at the bar's midpoint, symmetric about it, since "
        "heat generated there must conduct equally toward both cold ends."
    )


if __name__ == "__main__":
    main()
