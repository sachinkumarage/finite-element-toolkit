"""Example: a TET4 solid block cooled by convection, Version 21.

A single TET4 tetrahedron: one face held at a hot, fixed temperature,
the opposite (largest) face exposed to convective cooling, and the
remaining two faces left insulated. Demonstrates that the new
:mod:`femtoolkit.thermal.thermal_surfaces` machinery correctly resolves
one of TET4's four *faces* -- not just its four nodes -- and that the
result is a genuine 3D equilibrium between conduction and convection.
"""

from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Mesh, Node, Tet4Element3D
from femtoolkit.thermal import (
    PrescribedTemperature,
    SteadyStateThermalAnalysis,
    SteadyStateThermalResult,
    ThermalMaterial,
)
from femtoolkit.thermal.thermal_boundary_conditions import ConvectionBoundaryCondition
from femtoolkit.thermal.thermal_surfaces import (
    ThermalSurface,
    surface_area,
    surface_mean_temperature,
)

CONDUCTIVITY = 50.0  # W/(m*K)
DENSITY = 7850.0  # kg/m^3
SPECIFIC_HEAT = 460.0  # J/(kg*K)
HOT_TEMPERATURE = 500.0  # K
AMBIENT_TEMPERATURE = 300.0  # K
CONVECTION_COEFFICIENT = 30.0  # W/(m^2*K)


def main() -> None:
    """Build, solve, and report a convectively-cooled TET4 tetrahedron."""
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=DENSITY)
    nodes = (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=1.0, y=0.0, z=0.0),
        Node(id=3, x=0.0, y=1.0, z=0.0),
        Node(id=4, x=0.0, y=0.0, z=1.0),
    )
    tet = Tet4Element3D(id=1, nodes=nodes, material=placeholder)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(tet)

    thermal_material = ThermalMaterial(
        thermal_conductivity=CONDUCTIVITY, density=DENSITY, specific_heat=SPECIFIC_HEAT
    )
    analysis = SteadyStateThermalAnalysis(mesh, {tet.id: thermal_material})
    analysis.add_boundary_condition(PrescribedTemperature(nodes[0].id, HOT_TEMPERATURE))

    hypotenuse_face = ThermalSurface(tet.id, 3)  # the (1, 2, 3) face, opposite node 1
    analysis.add_convection(
        ConvectionBoundaryCondition(
            surface=hypotenuse_face,
            convection_coefficient=CONVECTION_COEFFICIENT,
            ambient_temperature=AMBIENT_TEMPERATURE,
        )
    )

    result = analysis.solve()
    print_summary(mesh, tet, hypotenuse_face, result)


def print_summary(
    mesh: Mesh, tet: Tet4Element3D, face: ThermalSurface, result: SteadyStateThermalResult
) -> None:
    """Print the solved nodal temperatures and the cooled face's convective flux."""
    print("Finite Element Toolkit")
    print("Version 21 -- TET4 Block Cooled by Convection")
    print("=" * 47)

    print(f"\nNode 1 fixed at {HOT_TEMPERATURE} K; face opposite it convects to ambient")
    print(f"h = {CONVECTION_COEFFICIENT} W/(m^2*K), T_infinity = {AMBIENT_TEMPERATURE} K")
    print(f"Convecting face area: {surface_area(mesh, face):.4f} m^2")

    print("\nNodal temperatures:")
    for node in tet.nodes:
        print(f"    Node {node.id} ({node.x:.2f},{node.y:.2f},{node.z:.2f}): "
              f"T={result.node_temperature(node.id):.4f} K")

    face_temperature = surface_mean_temperature(mesh, face, result.temperatures, result.dof_map)
    convective_flux = CONVECTION_COEFFICIENT * (face_temperature - AMBIENT_TEMPERATURE)
    print(f"\nMean cooled-face temperature: {face_temperature:.4f} K")
    print(f"Convective heat flux leaving that face: {convective_flux:.4f} W/m^2")


if __name__ == "__main__":
    main()
