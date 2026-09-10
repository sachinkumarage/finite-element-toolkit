"""Example: transient heat conduction via Backward Euler, Version 20.

A bar starts uniformly at room temperature; at t=0 one end is suddenly
held hot while the other stays cold. The temperature field evolves from
the flat initial condition toward the steady-state linear profile from
``one_dimensional_conduction.py``.

Two runs are shown. The first uses modest time steps to trace out the
early temperature evolution in physically meaningful increments. This
bar's thermal diffusivity, ``k/(rho*c)``, is small enough that reaching
true steady state this way would take hundreds of thousands of seconds
of simulated time -- so a second run demonstrates Backward Euler's
*unconditional* stability by jumping straight there with one deliberately
huge time step, converging onto the steady-state solver's own answer
with no overshoot or oscillation despite the enormous step.
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
    TransientThermalAnalysis,
    TransientThermalResult,
)

CONDUCTIVITY = 50.0  # W/(m*K)
DENSITY = 7850.0  # kg/m^3
SPECIFIC_HEAT = 460.0  # J/(kg*K)
LENGTH = 2.0  # m
AREA = 0.01  # m^2
NUM_ELEMENTS = 8
T_INITIAL = 293.15  # K
T_HOT = 373.15  # K
T_COLD = 293.15  # K
TIME_STEP = 20.0  # s
NUM_STEPS = 30
LONG_TIME_STEP = 1.0e7  # s, deliberately huge -- tests unconditional stability
LONG_NUM_STEPS = 5


def _build_mesh() -> tuple[Mesh, dict[int, ThermalMaterial], list[Node]]:
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
    return mesh, materials, nodes


def main() -> None:
    """Solve a transient heating problem and compare its long-time limit to steady state."""
    mesh, materials, nodes = _build_mesh()

    transient_analysis = TransientThermalAnalysis(
        mesh, materials, time_step=TIME_STEP, num_steps=NUM_STEPS, initial_temperature=T_INITIAL
    )
    transient_analysis.add_boundary_condition(PrescribedTemperature(nodes[0].id, T_HOT))
    transient_analysis.add_boundary_condition(PrescribedTemperature(nodes[-1].id, T_COLD))
    transient_result = transient_analysis.solve()

    long_time_analysis = TransientThermalAnalysis(
        mesh,
        materials,
        time_step=LONG_TIME_STEP,
        num_steps=LONG_NUM_STEPS,
        initial_temperature=T_INITIAL,
    )
    long_time_analysis.add_boundary_condition(PrescribedTemperature(nodes[0].id, T_HOT))
    long_time_analysis.add_boundary_condition(PrescribedTemperature(nodes[-1].id, T_COLD))
    long_time_result = long_time_analysis.solve()

    steady_state_analysis = SteadyStateThermalAnalysis(mesh, materials)
    steady_state_analysis.add_boundary_condition(PrescribedTemperature(nodes[0].id, T_HOT))
    steady_state_analysis.add_boundary_condition(PrescribedTemperature(nodes[-1].id, T_COLD))
    steady_state_result = steady_state_analysis.solve()

    print_summary(nodes, transient_result, long_time_result, steady_state_result)


def print_summary(
    nodes: list[Node],
    transient_result: TransientThermalResult,
    long_time_result: TransientThermalResult,
    steady_state_result: SteadyStateThermalResult,
) -> None:
    """Print the early temperature evolution and the long-time convergence to steady state."""
    print("Finite Element Toolkit")
    print("Version 20 -- Transient Heat Conduction (Backward Euler)")
    print("=" * 55)

    print(f"\nInitial temperature: {T_INITIAL} K, boundaries held at {T_HOT} K / {T_COLD} K")
    print(f"Time step: {TIME_STEP} s, steps: {NUM_STEPS}")

    midpoint_node = nodes[len(nodes) // 2]
    print(
        f"\nTemperature history at midpoint "
        f"(node {midpoint_node.id}, x={midpoint_node.x:.2f} m):"
    )
    history = transient_result.node_temperature_history(midpoint_node.id)
    sample_steps = sorted({0, 1, 2, 5, 10, 20, NUM_STEPS})
    for step in sample_steps:
        if step < len(history):
            print(f"    t={transient_result.times[step]:.1f} s: T={history[step]:.4f} K")

    print(
        f"\nHeat diffuses slowly here -- {NUM_STEPS * TIME_STEP:.0f} s of simulated time "
        "barely moves the midpoint temperature."
    )

    print(
        f"\nLong-time run: time step={LONG_TIME_STEP:.1e} s, steps={LONG_NUM_STEPS} "
        f"(total {LONG_TIME_STEP * LONG_NUM_STEPS:.1e} s)"
    )
    print("Final state vs. steady-state solver, at every node:")
    for node in nodes:
        long_time_final = long_time_result.node_temperature(node.id, step=-1)
        steady = steady_state_result.node_temperature(node.id)
        print(
            f"    x={node.x:.3f} m: transient={long_time_final:.4f} K, "
            f"steady-state={steady:.4f} K"
        )

    print(
        "\nDespite a single time step of ten million seconds, Backward Euler "
        "remains stable and converges directly onto the steady-state answer -- "
        "the long-time limit of the heat equation is the steady-state equation."
    )


if __name__ == "__main__":
    main()
