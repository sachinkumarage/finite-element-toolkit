"""Example: stress and strain visualization, Version 22.

Solves a cantilevered Q4 plate under a tip load (a classic bending
benchmark) and visualizes the resulting von Mises equivalent stress
field -- reusing the existing stress calculation
(:meth:`~femtoolkit.results.analysis_result.AnalysisResult.element_stress`)
entirely; this script performs no constitutive calculation of its own.
"""

from pathlib import Path

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.static_linear import StaticLinearAnalysis
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import create_quad_mesh
from femtoolkit.postprocessing import (
    PostProcessor,
    from_static_linear,
    plot_element_scatter_2d,
    with_derived_fields,
)

OUTPUT_PATH = Path(__file__).parent / "stress_visualization.png"


def main() -> None:
    """Solve a cantilever plate under a tip load and visualize von Mises stress."""
    material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    mesh = create_quad_mesh(width=2.0, height=0.4, nx=16, ny=4, material=material, thickness=0.02)

    analysis = StaticLinearAnalysis(mesh)
    for node in mesh.nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.X, 0.0))
            analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.Y, 0.0))
    for node in mesh.nodes:
        if node.x == 2.0 and node.y == 0.0:
            analysis.add_load(NodalLoad(node.id, TranslationDOF.Y, -5000.0))
    result = analysis.solve()

    simulation = with_derived_fields(from_static_linear(result))
    processor = PostProcessor(simulation)

    print("Finite Element Toolkit")
    print("Version 22 -- Stress and Strain Visualization")
    print("=" * 47)
    max_von_mises = processor.maximum("von_mises_stress", kind="element")
    print(f"\nMaximum von Mises stress: {max_von_mises:.4e} Pa")
    print(f"Maximum displacement:     {processor.maximum('displacement'):.6e} m")

    figure = plot_element_scatter_2d(simulation, "von_mises_stress")
    figure.savefig(OUTPUT_PATH, dpi=150)
    print(f"\nSaved von Mises stress plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
