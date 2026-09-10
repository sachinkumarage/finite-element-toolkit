"""Example: 2D steady-state heat conduction through a rectangular plate, Version 20.

Reuses the Version 8 structured Q4 mesh generator
(:func:`~femtoolkit.mesh.create_quad_mesh`) -- unmodified -- to build a
plate, then solves the same physics as
``one_dimensional_conduction.py`` embedded in two dimensions: the left
edge is held hot, the right edge cold, and the top/bottom edges are left
unconstrained (no boundary condition means no heat crosses them, i.e. a
perfectly insulated edge). Because nothing varies with y, the result is
the same linear T(x) profile as the 1D case -- demonstrating that Q4
thermal elements reduce correctly to the 1D solution when the physics is
effectively one-dimensional.
"""

from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, create_quad_mesh
from femtoolkit.thermal import (
    PrescribedTemperature,
    SteadyStateThermalAnalysis,
    SteadyStateThermalResult,
    ThermalMaterial,
)

CONDUCTIVITY = 50.0  # W/(m*K)
DENSITY = 7850.0  # kg/m^3
SPECIFIC_HEAT = 460.0  # J/(kg*K)
THICKNESS = 0.01  # m
WIDTH = 2.0  # m
HEIGHT = 1.0  # m
NX = 8
NY = 4
T_HOT = 373.15  # K
T_COLD = 293.15  # K


def main() -> None:
    """Build, solve, and report steady-state conduction through a Q4 plate."""
    placeholder_material = LinearElastic2D(
        youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress"
    )
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=NX, ny=NY, material=placeholder_material,
        thickness=THICKNESS,
    )

    thermal_material = ThermalMaterial(
        thermal_conductivity=CONDUCTIVITY, density=DENSITY, specific_heat=SPECIFIC_HEAT
    )
    materials = {element.id: thermal_material for element in mesh.elements}

    analysis = SteadyStateThermalAnalysis(mesh, materials)
    for node in mesh.nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(PrescribedTemperature(node.id, T_HOT))
        elif node.x == WIDTH:
            analysis.add_boundary_condition(PrescribedTemperature(node.id, T_COLD))

    result = analysis.solve()
    print_summary(mesh, result)


def print_summary(mesh: Mesh, result: SteadyStateThermalResult) -> None:
    """Print nodal temperatures along the plate's centerline against the 1D analytical solution."""
    print("Finite Element Toolkit")
    print("Version 20 -- 2D Steady-State Plate Conduction")
    print("=" * 50)

    print(f"\nPlate: {WIDTH} m x {HEIGHT} m, {NX}x{NY} Q4 elements, k={CONDUCTIVITY} W/(m*K)")
    print(f"Left edge (x=0): T={T_HOT} K, right edge (x={WIDTH}): T={T_COLD} K")
    print("Top/bottom edges: unconstrained (insulated -- no heat crosses them)")

    centerline_y = HEIGHT / 2.0
    centerline_nodes = sorted(
        (node for node in mesh.nodes if abs(node.y - centerline_y) < 1e-9), key=lambda n: n.x
    )
    print("\nCenterline temperatures (finite element vs. analytical T0 + (TL-T0)*x/W):")
    for node in centerline_nodes:
        analytical = T_HOT + (T_COLD - T_HOT) * node.x / WIDTH
        solved = result.node_temperature(node.id)
        print(f"    x={node.x:.3f} m: FE={solved:.4f} K, analytical={analytical:.4f} K")

    print(
        "\nBecause the boundary conditions do not vary with y, the 2D solution "
        "collapses to the same 1D linear profile everywhere -- a Q4 conduction "
        "consistency check against the simpler 1D bar case."
    )


if __name__ == "__main__":
    main()
