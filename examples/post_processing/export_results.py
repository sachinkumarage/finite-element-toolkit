"""Example: exporting results to CSV and JSON, Version 22.

Solves a heated, convectively-cooled plate and exports its temperature
and heat-flux fields to both supported file formats -- a CSV file (tidy,
one row per node/element/field/component observation, directly
importable into a spreadsheet) and a JSON file (the full result,
every step, nested by topology/step/field).
"""

from pathlib import Path

from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import create_quad_mesh
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.postprocessing import (
    export_to_csv,
    export_to_json,
    from_thermal_steady_state,
    with_derived_fields,
)
from femtoolkit.thermal import PrescribedTemperature, SteadyStateThermalAnalysis, ThermalMaterial
from femtoolkit.thermal.thermal_boundary_conditions import ConvectionBoundaryCondition
from femtoolkit.thermal.thermal_surfaces import ThermalSurface

CSV_OUTPUT = Path(__file__).parent / "export_results.csv"
JSON_OUTPUT = Path(__file__).parent / "export_results.json"

_QUAD_LOCAL_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))


def _find_right_edge_surface(mesh: Mesh, width: float) -> ThermalSurface:
    """Find the (element, local edge index) whose edge lies entirely on ``x = width``."""
    for element in mesh.elements:
        for edge_index, (i, j) in enumerate(_QUAD_LOCAL_EDGES):
            node_a, node_b = element.nodes[i], element.nodes[j]
            if node_a.x == width and node_b.x == width:
                return ThermalSurface(element.id, edge_index)
    raise ValueError(f"No element edge found on x={width}.")


def main() -> None:
    """Solve a heated plate and export its results to CSV and JSON."""
    placeholder_material = LinearElastic2D(
        youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress"
    )
    mesh = create_quad_mesh(
        width=1.0, height=0.5, nx=4, ny=2, material=placeholder_material, thickness=0.01
    )
    thermal_material = ThermalMaterial(
        thermal_conductivity=50.0, density=7850.0, specific_heat=460.0
    )
    materials = {element.id: thermal_material for element in mesh.elements}
    analysis = SteadyStateThermalAnalysis(mesh, materials)
    for node in mesh.nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(PrescribedTemperature(node.id, 400.0))

    surface = _find_right_edge_surface(mesh, width=1.0)
    analysis.add_convection(
        ConvectionBoundaryCondition(
            surface=surface, convection_coefficient=25.0, ambient_temperature=293.15
        )
    )
    result = analysis.solve()

    simulation = with_derived_fields(from_thermal_steady_state(result))

    export_to_csv(simulation, CSV_OUTPUT)
    export_to_json(simulation, JSON_OUTPUT)

    print("Finite Element Toolkit")
    print("Version 22 -- Exporting Results to CSV and JSON")
    print("=" * 50)
    print(f"\nExported {len(simulation.topology.node_ids)} nodes and "
          f"{len(simulation.topology.element_ids)} elements.")
    print(f"CSV saved to:  {CSV_OUTPUT}")
    print(f"JSON saved to: {JSON_OUTPUT}")

    with CSV_OUTPUT.open(encoding="utf-8") as csv_file:
        preview_lines = [next(csv_file) for _ in range(4)]
    print("\nCSV preview (first 4 lines):")
    for line in preview_lines:
        print(f"    {line.rstrip()}")


if __name__ == "__main__":
    main()
