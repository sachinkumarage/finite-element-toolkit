"""Example: heat-flux visualization, Version 22.

Solves the same convection-cooled plate as ``temperature_contour.py``
and visualizes its heat flux two ways: a filled contour of flux
magnitude, and a vector (quiver) plot showing flux direction -- heat
flowing from the hot edge toward the cooled edge, exactly as Fourier's
law predicts.
"""

from pathlib import Path

from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import create_quad_mesh
from femtoolkit.postprocessing import (
    PostProcessor,
    from_thermal_steady_state,
    plot_element_scatter_2d,
    plot_heat_flux_vectors_2d,
    with_derived_fields,
)
from femtoolkit.thermal import PrescribedTemperature, SteadyStateThermalAnalysis, ThermalMaterial

MAGNITUDE_OUTPUT = Path(__file__).parent / "heat_flux_magnitude.png"
VECTOR_OUTPUT = Path(__file__).parent / "heat_flux_vectors.png"


def main() -> None:
    """Solve a heated plate and visualize its heat flux magnitude and direction."""
    placeholder_material = LinearElastic2D(
        youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress"
    )
    mesh = create_quad_mesh(
        width=2.0, height=1.0, nx=8, ny=4, material=placeholder_material, thickness=0.01
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

    simulation = with_derived_fields(from_thermal_steady_state(result))
    processor = PostProcessor(simulation)

    print("Finite Element Toolkit")
    print("Version 22 -- Heat Flux Visualization")
    print("=" * 40)
    max_flux = processor.maximum("heat_flux_magnitude", kind="element")
    print(f"\nMaximum heat flux magnitude: {max_flux:.2f} W/m^2")

    figure_magnitude = plot_element_scatter_2d(simulation, "heat_flux_magnitude")
    figure_magnitude.savefig(MAGNITUDE_OUTPUT, dpi=150)
    print(f"Saved heat flux magnitude plot to {MAGNITUDE_OUTPUT}")

    figure_vectors = plot_heat_flux_vectors_2d(simulation)
    figure_vectors.savefig(VECTOR_OUTPUT, dpi=150)
    print(f"Saved heat flux vector plot to {VECTOR_OUTPUT}")


if __name__ == "__main__":
    main()
