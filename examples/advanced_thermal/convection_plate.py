"""Example: a heated plate exposed to air, Version 21.

Reuses the Version 8 structured Q4 mesh generator to build a plate whose
left edge is held at a hot, fixed temperature while every other edge is
exposed to ambient air by convection (Newton's law of cooling,
``q = h*(T - T_infinity)``). Unlike ``examples/heat_transfer/steady_state_plate.py``
(Version 20, insulated top/bottom edges), heat here can escape through
three of the plate's four edges -- a much more realistic "plate in a
room" boundary condition -- so the resulting temperature field is
genuinely two-dimensional rather than a disguised 1D profile.
"""

from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, create_quad_mesh
from femtoolkit.mesh.node import Node
from femtoolkit.mesh.quad_element import QuadElement2D
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
THICKNESS = 0.01  # m
WIDTH = 2.0  # m
HEIGHT = 1.0  # m
NX = 8
NY = 4
HOT_TEMPERATURE = 400.0  # K
AMBIENT_TEMPERATURE = 293.15  # K
CONVECTION_COEFFICIENT = 15.0  # W/(m^2*K), still air


def main() -> None:
    """Build, solve, and report a convectively-cooled plate."""
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
    left_edge_nodes = [node for node in mesh.nodes if node.x == 0.0]
    for node in left_edge_nodes:
        analysis.add_boundary_condition(PrescribedTemperature(node.id, HOT_TEMPERATURE))

    exposed_edges = 0
    for element in mesh.elements:
        for edge_index in range(4):
            node_a, node_b = _edge_nodes(element, edge_index)
            if node_a.x == 0.0 and node_b.x == 0.0:
                continue  # the hot (Dirichlet) edge is not exposed to air
            if _is_boundary_edge(node_a, node_b, WIDTH, HEIGHT):
                analysis.add_convection(
                    ConvectionBoundaryCondition(
                        surface=ThermalSurface(element.id, edge_index),
                        convection_coefficient=CONVECTION_COEFFICIENT,
                        ambient_temperature=AMBIENT_TEMPERATURE,
                    )
                )
                exposed_edges += 1

    result = analysis.solve()
    print_summary(mesh, result, exposed_edges)


def _edge_nodes(element: QuadElement2D, edge_index: int) -> tuple[Node, Node]:
    local_edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    i, j = local_edges[edge_index]
    return element.nodes[i], element.nodes[j]


def _is_boundary_edge(node_a: Node, node_b: Node, width: float, height: float) -> bool:
    on_top = node_a.y == height and node_b.y == height
    on_bottom = node_a.y == 0.0 and node_b.y == 0.0
    on_right = node_a.x == width and node_b.x == width
    return on_top or on_bottom or on_right


def print_summary(mesh: Mesh, result: SteadyStateThermalResult, exposed_edges: int) -> None:
    """Print the solved temperature field along the plate's centerline."""
    print("Finite Element Toolkit")
    print("Version 21 -- Heated Plate Exposed to Air")
    print("=" * 45)

    print(f"\nPlate: {WIDTH} m x {HEIGHT} m, {NX}x{NY} Q4 elements")
    print(f"Left edge fixed at {HOT_TEMPERATURE} K; {exposed_edges} boundary edges exposed to air")
    print(
        f"Convection coefficient: {CONVECTION_COEFFICIENT} W/(m^2*K), "
        f"ambient: {AMBIENT_TEMPERATURE} K"
    )

    centerline_y = HEIGHT / 2.0
    centerline_nodes = sorted(
        (node for node in mesh.nodes if abs(node.y - centerline_y) < 1e-9), key=lambda n: n.x
    )
    print("\nCenterline temperatures:")
    for node in centerline_nodes:
        print(f"    x={node.x:.3f} m: T={result.node_temperature(node.id):.4f} K")

    print(
        "\nUnlike an insulated-edge plate, heat now escapes through the top, "
        "bottom, and right edges too -- the temperature falls off faster "
        "than the simple 1D profile a fully-insulated plate would show."
    )


if __name__ == "__main__":
    main()
