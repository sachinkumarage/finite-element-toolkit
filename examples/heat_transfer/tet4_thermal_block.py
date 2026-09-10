"""Example: TET4 solid heat conduction, Version 20.

A single TET4 tetrahedron under an imposed linear temperature field --
the thermal analogue of the mechanical TET4 patch test
(``tests/validation/test_tet4_patch.py``). TET4's shape functions are
linear, so a linear temperature field, its gradient, and its heat flux
should all be reproduced *exactly*, at any orientation. This is the
toolkit's TET4 thermal patch test, demonstrated interactively here.
"""

import numpy as np

from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Mesh, Node, Tet4Element3D
from femtoolkit.thermal import (
    PrescribedTemperature,
    SteadyStateThermalAnalysis,
    SteadyStateThermalResult,
    ThermalMaterial,
)

CONDUCTIVITY = 50.0  # W/(m*K)
DENSITY = 7850.0  # kg/m^3
SPECIFIC_HEAT = 460.0  # J/(kg*K)
T_REFERENCE = 293.15  # K
IMPOSED_GRADIENT = np.array([40.0, 0.0, 0.0])  # K/m


def main() -> None:
    """Build, solve, and verify a TET4 thermal patch test."""
    placeholder_material = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=DENSITY)
    nodes = (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=1.0, y=0.0, z=0.0),
        Node(id=3, x=0.3, y=1.0, z=0.1),
        Node(id=4, x=0.2, y=0.2, z=1.0),
    )
    tet = Tet4Element3D(id=1, nodes=nodes, material=placeholder_material)

    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(tet)

    thermal_material = ThermalMaterial(
        thermal_conductivity=CONDUCTIVITY, density=DENSITY, specific_heat=SPECIFIC_HEAT
    )
    analysis = SteadyStateThermalAnalysis(mesh, {tet.id: thermal_material})
    for node in nodes:
        position = np.array([node.x, node.y, node.z])
        temperature = T_REFERENCE + float(IMPOSED_GRADIENT @ position)
        analysis.add_boundary_condition(PrescribedTemperature(node.id, temperature))

    result = analysis.solve()
    print_summary(tet, nodes, result)


def print_summary(
    tet: Tet4Element3D, nodes: tuple[Node, ...], result: SteadyStateThermalResult
) -> None:
    """Print the solved nodal temperatures, recovered gradient, and heat flux."""
    print("Finite Element Toolkit")
    print("Version 20 -- TET4 Solid Heat Conduction")
    print("=" * 45)

    print(f"\nImposed gradient: {IMPOSED_GRADIENT} K/m, T(0,0,0) = {T_REFERENCE} K")
    print("\nNodal temperatures (finite element vs. imposed linear field):")
    for node in nodes:
        position = np.array([node.x, node.y, node.z])
        expected = T_REFERENCE + float(IMPOSED_GRADIENT @ position)
        solved = result.node_temperature(node.id)
        print(f"    Node {node.id} ({node.x:.2f},{node.y:.2f},{node.z:.2f}): "
              f"FE={solved:.4f} K, expected={expected:.4f} K")

    gradient = result.element_temperature_gradient(tet.id)
    print(f"\nRecovered temperature gradient: {gradient} K/m (imposed: {IMPOSED_GRADIENT} K/m)")

    flux = result.element_heat_flux(tet.id)
    expected_flux = -CONDUCTIVITY * IMPOSED_GRADIENT
    print(f"Recovered heat flux, q=-k*grad(T): {flux} W/m^2 (expected: {expected_flux} W/m^2)")

    print(
        "\nA linear TET4 element reproduces an exactly linear temperature field, "
        "its constant gradient, and its constant heat flux everywhere -- the "
        "thermal patch test."
    )


if __name__ == "__main__":
    main()
