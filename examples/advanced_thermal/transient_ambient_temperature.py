"""Example: furnace heating with a time-dependent ambient temperature, Version 21.

A block starts at room temperature, exposed on one face to a furnace
whose ambient temperature suddenly steps up partway through the
simulation (reusing
:class:`~femtoolkit.analysis.dynamic_loads.StepLoad`, the same
time-history shape already used for time-varying mechanical loads). The
opposite face is held fixed at room temperature throughout.
:class:`~femtoolkit.thermal.thermal_analysis.TransientThermalAnalysis`
re-resolves the ambient temperature at every Backward-Euler time step,
so the exposed face only starts heating once the furnace switches on.
"""

from femtoolkit.analysis.dynamic_loads import StepLoad
from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.thermal import (
    PrescribedTemperature,
    ThermalMaterial,
    TransientThermalAnalysis,
    TransientThermalResult,
)
from femtoolkit.thermal.thermal_boundary_conditions import ConvectionBoundaryCondition
from femtoolkit.thermal.thermal_surfaces import ThermalSurface

CONDUCTIVITY = 25.0  # W/(m*K)
DENSITY = 7850.0  # kg/m^3
SPECIFIC_HEAT = 460.0  # J/(kg*K)
ROOM_TEMPERATURE = 293.15  # K
FURNACE_TEMPERATURE = 900.0  # K
FURNACE_START_TIME = 6000.0  # s
CONVECTION_COEFFICIENT = 40.0  # W/(m^2*K)
RIGHT_FACE_INDEX = 4  # xi = +1, the x = 1 face -- see HEX8_FACES
TIME_STEP = 1000.0  # s
NUM_STEPS = 30

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def main() -> None:
    """Build, solve, and report a block exposed to a furnace that switches on partway through."""
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
    furnace_cycle = StepLoad(magnitude=FURNACE_TEMPERATURE, step_time=FURNACE_START_TIME)

    analysis = TransientThermalAnalysis(
        mesh,
        {hexa.id: thermal_material},
        time_step=TIME_STEP,
        num_steps=NUM_STEPS,
        initial_temperature=ROOM_TEMPERATURE,
    )
    for node in nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(PrescribedTemperature(node.id, ROOM_TEMPERATURE))
    right_face = ThermalSurface(hexa.id, RIGHT_FACE_INDEX)
    analysis.add_convection(
        ConvectionBoundaryCondition(
            surface=right_face,
            convection_coefficient=CONVECTION_COEFFICIENT,
            ambient_temperature=furnace_cycle,
        )
    )

    result = analysis.solve()
    print_summary(result)


def print_summary(result: TransientThermalResult) -> None:
    """Print the exposed face's temperature history around the furnace switch-on time."""
    print("Finite Element Toolkit")
    print("Version 21 -- Furnace Heating (Time-Dependent Ambient Temperature)")
    print("=" * 68)

    print(f"\nRoom temperature: {ROOM_TEMPERATURE} K")
    print(f"Furnace switches on to {FURNACE_TEMPERATURE} K at t={FURNACE_START_TIME} s")

    history = result.node_temperature_history(2)
    print("\nExposed-face (node 2) temperature history:")
    for step, time in enumerate(result.times):
        marker = "  <- furnace on" if abs(time - FURNACE_START_TIME) < 1e-9 else ""
        print(f"    t={time:6.1f} s: T={history[step]:.4f} K{marker}")

    print(
        "\nThe exposed face stays near room temperature until the furnace "
        "switches on, then climbs toward the new, much hotter ambient "
        "condition -- the ambient temperature is re-resolved at every time step."
    )


if __name__ == "__main__":
    main()
