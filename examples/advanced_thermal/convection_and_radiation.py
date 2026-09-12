"""Example: combined convection and radiation on the same surface, Version 21.

A hot surface loses heat through *both* mechanisms at once -- exactly
the realistic situation for, say, a furnace wall exposed to room air:
convection carries heat away through the surrounding fluid, while
radiation carries heat away independently through electromagnetic
emission. Both boundary conditions can be added to the same surface;
the nonlinear thermal solver accounts for their combined effect in a
single Newton-Raphson solve. This script compares the equilibrium
reached with convection alone, radiation alone, and both together, to
show the combined case cools the surface further than either alone.
"""

from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.thermal import PrescribedTemperature, SteadyStateThermalAnalysis, ThermalMaterial
from femtoolkit.thermal.thermal_boundary_conditions import (
    ConvectionBoundaryCondition,
    RadiationBoundaryCondition,
)
from femtoolkit.thermal.thermal_surfaces import ThermalSurface

CONDUCTIVITY = 25.0  # W/(m*K)
DENSITY = 7850.0  # kg/m^3
SPECIFIC_HEAT = 460.0  # J/(kg*K)
HOT_TEMPERATURE = 700.0  # K
AMBIENT_TEMPERATURE = 300.0  # K
CONVECTION_COEFFICIENT = 20.0  # W/(m^2*K)
EMISSIVITY = 0.8
RIGHT_FACE_INDEX = 4  # xi = +1, the x = 1 face -- see HEX8_FACES

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def _build_block() -> tuple[Mesh, Hex8Element3D]:
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=DENSITY)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)
    return mesh, hexa


def _solve(with_convection: bool, with_radiation: bool) -> float:
    mesh, hexa = _build_block()
    thermal_material = ThermalMaterial(
        thermal_conductivity=CONDUCTIVITY, density=DENSITY, specific_heat=SPECIFIC_HEAT
    )
    analysis = SteadyStateThermalAnalysis(mesh, {hexa.id: thermal_material})
    for node in hexa.nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(PrescribedTemperature(node.id, HOT_TEMPERATURE))

    right_face = ThermalSurface(hexa.id, RIGHT_FACE_INDEX)
    if with_convection:
        analysis.add_convection(
            ConvectionBoundaryCondition(
                surface=right_face,
                convection_coefficient=CONVECTION_COEFFICIENT,
                ambient_temperature=AMBIENT_TEMPERATURE,
            )
        )
    if with_radiation:
        analysis.add_radiation(
            RadiationBoundaryCondition(
                surface=right_face,
                emissivity=EMISSIVITY,
                surrounding_temperature=AMBIENT_TEMPERATURE,
            )
        )

    result = analysis.solve()
    return result.node_temperature(2)


def main() -> None:
    """Compare the exposed-face equilibrium temperature under each cooling combination."""
    convection_only = _solve(with_convection=True, with_radiation=False)
    radiation_only = _solve(with_convection=False, with_radiation=True)
    both = _solve(with_convection=True, with_radiation=True)

    print("Finite Element Toolkit")
    print("Version 21 -- Combined Convection and Radiation")
    print("=" * 48)

    print(f"\nHot face: {HOT_TEMPERATURE} K, ambient/surroundings: {AMBIENT_TEMPERATURE} K")
    print(f"h = {CONVECTION_COEFFICIENT} W/(m^2*K), emissivity = {EMISSIVITY}")

    print("\nExposed-face equilibrium temperature:")
    print(f"    Convection alone:  {convection_only:.4f} K")
    print(f"    Radiation alone:   {radiation_only:.4f} K")
    print(f"    Both combined:     {both:.4f} K")

    assert both < min(convection_only, radiation_only)
    print(
        "\nWith both mechanisms active, the surface loses heat faster than "
        "with either alone, so it settles at a lower equilibrium temperature."
    )


if __name__ == "__main__":
    main()
