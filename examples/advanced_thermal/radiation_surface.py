"""Example: a radiating surface, Version 21.

A HEX8 unit cube: one face held at a hot, fixed temperature, the
opposite face radiating to cold surroundings via the Stefan-Boltzmann
law (``q = epsilon*sigma*(T^4 - T_surroundings^4)``). Because radiation
depends on ``T^4``, the thermal problem is genuinely nonlinear -- solved
here by the Newton-Raphson iteration
:mod:`femtoolkit.thermal.thermal_analysis` switches to automatically
whenever a radiation boundary condition is present.
"""

from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.thermal import (
    PrescribedTemperature,
    SteadyStateThermalAnalysis,
    SteadyStateThermalResult,
    ThermalMaterial,
)
from femtoolkit.thermal.thermal_boundary_conditions import (
    STEFAN_BOLTZMANN_CONSTANT,
    RadiationBoundaryCondition,
    radiative_heat_flux,
)
from femtoolkit.thermal.thermal_surfaces import ThermalSurface, surface_mean_temperature

CONDUCTIVITY = 20.0  # W/(m*K)
DENSITY = 7850.0  # kg/m^3
SPECIFIC_HEAT = 460.0  # J/(kg*K)
HOT_TEMPERATURE = 800.0  # K
SURROUNDING_TEMPERATURE = 300.0  # K
EMISSIVITY = 0.85
RIGHT_FACE_INDEX = 4  # xi = +1, the x = 1 face -- see HEX8_FACES

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def main() -> None:
    """Build, solve, and report a radiating HEX8 unit cube."""
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=DENSITY)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    thermal_material = ThermalMaterial(
        thermal_conductivity=CONDUCTIVITY, density=DENSITY, specific_heat=SPECIFIC_HEAT
    )
    analysis = SteadyStateThermalAnalysis(mesh, {hexa.id: thermal_material})
    for node in nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(PrescribedTemperature(node.id, HOT_TEMPERATURE))
    right_face = ThermalSurface(hexa.id, RIGHT_FACE_INDEX)
    analysis.add_radiation(
        RadiationBoundaryCondition(
            surface=right_face,
            emissivity=EMISSIVITY,
            surrounding_temperature=SURROUNDING_TEMPERATURE,
        )
    )

    result = analysis.solve()
    print_summary(mesh, nodes, right_face, result)


def print_summary(
    mesh: Mesh, nodes: tuple[Node, ...], face: ThermalSurface, result: SteadyStateThermalResult
) -> None:
    """Print the solved temperatures and the radiative heat flux leaving the cooled face."""
    print("Finite Element Toolkit")
    print("Version 21 -- Radiating Surface")
    print("=" * 35)

    print(f"\nx=0 face fixed at {HOT_TEMPERATURE} K")
    print(
        f"x=1 face radiates to surroundings at {SURROUNDING_TEMPERATURE} K, "
        f"emissivity={EMISSIVITY}"
    )
    print(f"Stefan-Boltzmann constant: {STEFAN_BOLTZMANN_CONSTANT:.6e} W/(m^2*K^4)")

    print("\nNodal temperatures:")
    for node in nodes:
        print(f"    Node {node.id} (x={node.x}): T={result.node_temperature(node.id):.4f} K")

    face_temperature = surface_mean_temperature(mesh, face, result.temperatures, result.dof_map)
    flux = radiative_heat_flux(EMISSIVITY, face_temperature, SURROUNDING_TEMPERATURE)
    print(f"\nRadiating face mean temperature: {face_temperature:.4f} K")
    print(f"Radiative heat flux leaving that face: {flux:.4f} W/m^2")

    print(
        "\nSolved via Newton-Raphson: radiation's T^4 dependence makes the "
        "thermal problem nonlinear, unlike constant-coefficient conduction "
        "or convection."
    )


if __name__ == "__main__":
    main()
