"""Example: temperature contour visualization, Version 22.

Solves a heated plate (Version 20's convection-cooled plate scenario)
and renders its temperature field as a filled 2D contour plot -- the
first, most basic post-processing task: turning a table of nodal
temperatures into a picture an engineer can read at a glance. Saves the
figure to a PNG file rather than opening an interactive window (this
version deliberately builds no GUI).
"""

from pathlib import Path

from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import create_quad_mesh
from femtoolkit.postprocessing import (
    PostProcessor,
    from_thermal_steady_state,
    plot_nodal_contour_2d,
)
from femtoolkit.thermal import PrescribedTemperature, SteadyStateThermalAnalysis, ThermalMaterial

OUTPUT_PATH = Path(__file__).parent / "temperature_contour.png"


def main() -> None:
    """Solve a heated plate and render its temperature field as a contour plot."""
    placeholder_material = LinearElastic2D(
        youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress"
    )
    mesh = create_quad_mesh(
        width=2.0, height=1.0, nx=12, ny=6, material=placeholder_material, thickness=0.01
    )

    thermal_material = ThermalMaterial(
        thermal_conductivity=50.0, density=7850.0, specific_heat=460.0
    )
    materials = {element.id: thermal_material for element in mesh.elements}
    analysis = SteadyStateThermalAnalysis(mesh, materials)
    for node in mesh.nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(PrescribedTemperature(node.id, 400.0))
        elif node.x == 2.0:
            analysis.add_boundary_condition(PrescribedTemperature(node.id, 293.15))
    result = analysis.solve()

    simulation = from_thermal_steady_state(result)
    processor = PostProcessor(simulation)

    print("Finite Element Toolkit")
    print("Version 22 -- Temperature Contour Visualization")
    print("=" * 50)
    print(f"\nMinimum temperature: {processor.minimum('temperature'):.2f} K")
    print(f"Maximum temperature: {processor.maximum('temperature'):.2f} K")
    print(f"Mean temperature:    {processor.mean('temperature'):.2f} K")

    figure = plot_nodal_contour_2d(simulation, "temperature")
    figure.savefig(OUTPUT_PATH, dpi=150)
    print(f"\nSaved contour plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
