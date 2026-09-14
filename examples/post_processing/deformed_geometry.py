"""Example: deformed geometry visualization, Version 22.

Solves the same cantilever plate as ``stress_visualization.py`` and
plots its **deformed** shape -- ``x_deformed = x_original + s*u`` --
alongside the original, undeformed outline. The true tip deflection is a
fraction of a millimeter on a 2-meter plate, invisible on a 1:1 plot, so
a visualization scale factor ``s`` (purely cosmetic -- see
:func:`~femtoolkit.postprocessing.field_calculator.deformed_coordinates`)
exaggerates it enough to see.
"""

from pathlib import Path

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.static_linear import StaticLinearAnalysis
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import create_quad_mesh
from femtoolkit.postprocessing import (
    PostProcessor,
    from_static_linear,
    plot_deformed_shape_2d,
    with_derived_fields,
)

VISUALIZATION_SCALE = 300.0
OUTPUT_PATH = Path(__file__).parent / "deformed_geometry.png"


def main() -> None:
    """Solve a cantilever plate and plot its (visually scaled) deformed shape."""
    material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    mesh = create_quad_mesh(width=2.0, height=0.4, nx=10, ny=2, material=material, thickness=0.02)

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
    print("Version 22 -- Deformed Geometry Visualization")
    print("=" * 47)
    print(f"\nTrue maximum displacement: {processor.maximum('displacement'):.6e} m")
    print(f"Visualization scale factor: {VISUALIZATION_SCALE:g} (cosmetic only)")

    figure = plot_deformed_shape_2d(
        simulation, scale=VISUALIZATION_SCALE, color_field="von_mises_stress"
    )
    figure.savefig(OUTPUT_PATH, dpi=150)
    print(f"\nSaved deformed shape plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
