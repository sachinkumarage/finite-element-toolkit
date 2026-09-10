"""Example: HEX8 solid heat conduction, Version 20.

A single unit-cube HEX8 element embeds the same 1D conduction problem
as ``one_dimensional_conduction.py``: one face held hot, the opposite
face held cold, and the remaining four faces unconstrained (insulated).
Confirms HEX8's trilinear shape functions reproduce the exact 1D linear
profile and heat flux along the conduction direction, with zero flux in
the other two directions.
"""

from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.thermal import (
    PrescribedTemperature,
    SteadyStateThermalAnalysis,
    SteadyStateThermalResult,
    ThermalMaterial,
)

CONDUCTIVITY = 50.0  # W/(m*K)
DENSITY = 7850.0  # kg/m^3
SPECIFIC_HEAT = 460.0  # J/(kg*K)
T_HOT = 373.15  # K
T_COLD = 293.15  # K

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def main() -> None:
    """Build, solve, and report a unit-cube HEX8 conduction problem."""
    placeholder_material = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=DENSITY)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder_material)

    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    thermal_material = ThermalMaterial(
        thermal_conductivity=CONDUCTIVITY, density=DENSITY, specific_heat=SPECIFIC_HEAT
    )
    analysis = SteadyStateThermalAnalysis(mesh, {hexa.id: thermal_material})
    for node in nodes:
        analysis.add_boundary_condition(
            PrescribedTemperature(node.id, T_HOT if node.x == 0.0 else T_COLD)
        )

    result = analysis.solve()
    print_summary(hexa, nodes, result)


def print_summary(
    hexa: Hex8Element3D, nodes: tuple[Node, ...], result: SteadyStateThermalResult
) -> None:
    """Print the solved nodal temperatures and heat flux against the 1D analytical case."""
    print("Finite Element Toolkit")
    print("Version 20 -- HEX8 Solid Heat Conduction")
    print("=" * 45)

    print(f"\nUnit cube, x=0 face at {T_HOT} K, x=1 face at {T_COLD} K, k={CONDUCTIVITY} W/(m*K)")
    print("\nNodal temperatures (finite element vs. analytical T0 + (TL-T0)*x):")
    for node in nodes:
        analytical = T_HOT + (T_COLD - T_HOT) * node.x
        solved = result.node_temperature(node.id)
        print(f"    Node {node.id} (x={node.x}): FE={solved:.4f} K, analytical={analytical:.4f} K")

    flux = result.element_heat_flux(hexa.id)
    expected_flux_x = -CONDUCTIVITY * (T_COLD - T_HOT)
    print(f"\nHeat flux: {flux} W/m^2 (expected q_x={expected_flux_x:.4f}, q_y=q_z=0)")

    print(
        "\nHEX8's trilinear shape functions reproduce the 1D linear temperature "
        "profile exactly, with heat flowing only along the conduction direction."
    )


if __name__ == "__main__":
    main()
