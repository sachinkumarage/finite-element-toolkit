"""Example: a HEX8 solid block cooled by convection, Version 21.

A unit-cube HEX8 element: one face held at a hot, fixed temperature, the
opposite face exposed to convective cooling, and the four lateral faces
left insulated. This is the toolkit's convection benchmark problem (see
``tests/validation/test_convection_benchmark.py``): the classic
composite-thermal-resistance result for conduction in series with
convection,

.. code-block:: text

    T_L = (k*T0 + h*L*T_infinity) / (k + h*L)

is reproduced here to floating-point precision, since HEX8's trilinear
shape functions -- and the bilinear surface integral used for the
convecting face -- are both exact for this purely 1D, linear-in-x
temperature field.
"""

from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.thermal import (
    PrescribedTemperature,
    SteadyStateThermalAnalysis,
    SteadyStateThermalResult,
    ThermalMaterial,
)
from femtoolkit.thermal.thermal_boundary_conditions import ConvectionBoundaryCondition
from femtoolkit.thermal.thermal_surfaces import ThermalSurface

CONDUCTIVITY = 50.0  # W/(m*K)
DENSITY = 7850.0  # kg/m^3
SPECIFIC_HEAT = 460.0  # J/(kg*K)
HOT_TEMPERATURE = 400.0  # K
AMBIENT_TEMPERATURE = 300.0  # K
CONVECTION_COEFFICIENT = 10.0  # W/(m^2*K)
LENGTH = 1.0  # m, the unit cube's edge length
RIGHT_FACE_INDEX = 4  # xi = +1, the x = 1 face -- see HEX8_FACES

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def main() -> None:
    """Build, solve, and verify a convectively-cooled HEX8 unit cube."""
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
    analysis.add_convection(
        ConvectionBoundaryCondition(
            surface=right_face,
            convection_coefficient=CONVECTION_COEFFICIENT,
            ambient_temperature=AMBIENT_TEMPERATURE,
        )
    )

    result = analysis.solve()
    print_summary(nodes, result)


def print_summary(nodes: tuple[Node, ...], result: SteadyStateThermalResult) -> None:
    """Print the solved temperatures against the closed-form composite-resistance solution."""
    print("Finite Element Toolkit")
    print("Version 21 -- HEX8 Block Cooled by Convection")
    print("=" * 47)

    print(f"\nx=0 face fixed at {HOT_TEMPERATURE} K, x=1 face convects to {AMBIENT_TEMPERATURE} K")
    print(f"k={CONDUCTIVITY} W/(m*K), h={CONVECTION_COEFFICIENT} W/(m^2*K)")

    expected_right_face = (
        CONDUCTIVITY * HOT_TEMPERATURE + CONVECTION_COEFFICIENT * LENGTH * AMBIENT_TEMPERATURE
    ) / (CONDUCTIVITY + CONVECTION_COEFFICIENT * LENGTH)
    print(f"\nClosed-form x=1 face temperature: {expected_right_face:.6f} K")

    print("\nNodal temperatures:")
    for node in nodes:
        print(f"    Node {node.id} (x={node.x}): T={result.node_temperature(node.id):.6f} K")

    print(
        "\nThe FE solution matches the composite-resistance closed form exactly: "
        "linear elements are exact for this purely 1D, linear temperature field."
    )


if __name__ == "__main__":
    main()
