"""Example: uniform and local mesh refinement, Version 25.

Generates a coarse structured mesh, uniformly refines it (edge-midpoint
quadrisection), and locally refines a single element of the original
mesh, comparing node/element counts, total area (conserved exactly),
and shape quality before and after -- reusing
:func:`~femtoolkit.mesh.refinement.refine_uniform`, the same function
the GUI's Mesh page (Model Preparation -> Refine Mesh) calls. Every
refined mesh is validated automatically by the refiner itself before
being returned.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt

from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh.generator import create_quad_mesh
from femtoolkit.mesh.quality import QualityEvaluator
from femtoolkit.mesh.refinement import refine_uniform
from femtoolkit.mesh.validation import generate_validation_report

OUTPUT_PATH = Path(__file__).parent / "mesh_refinement_demo.png"


def _plot_mesh(axis, mesh, title: str) -> None:
    for element in mesh.elements:
        xs = [node.x for node in element.nodes] + [element.nodes[0].x]
        ys = [node.y for node in element.nodes] + [element.nodes[0].y]
        axis.plot(xs, ys, "-", color="tab:gray", linewidth=0.6)
    for node in mesh.nodes:
        axis.plot(node.x, node.y, "o", color="tab:blue", markersize=2)
    axis.set_aspect("equal")
    axis.set_title(title)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")


def main() -> None:
    """Refine a coarse mesh uniformly and locally, reporting the results."""
    material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    evaluator = QualityEvaluator()

    print("Finite Element Toolkit")
    print("Version 25 -- Mesh Refinement")
    print("=" * 30)

    coarse = create_quad_mesh(width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01)
    coarse_area = sum(element.area for element in coarse.elements)
    print(f"\nCoarse mesh: {len(coarse.nodes)} nodes, {len(coarse.elements)} elements")
    print(f"  Total area: {coarse_area:.6f} m^2")

    uniform = refine_uniform(coarse)
    uniform_area = sum(element.area for element in uniform.elements)
    uniform_report = evaluator.evaluate(uniform)
    print(f"\nUniformly refined: {len(uniform.nodes)} nodes, {len(uniform.elements)} elements")
    area_conserved = math.isclose(uniform_area, coarse_area, rel_tol=1e-9)
    print(f"  Total area: {uniform_area:.6f} m^2 (conserved: {area_conserved})")
    print(
        f"  Quality: min={uniform_report.minimum_quality:.3f}, "
        f"mean={uniform_report.mean_quality:.3f}"
    )

    uniform_validation = generate_validation_report(uniform)
    print(f"  Validation status: {uniform_validation.status}")

    local = refine_uniform(coarse, elements=[1])
    local_area = sum(element.area for element in local.elements)
    print(
        f"\nLocally refined (element 1 only): {len(local.nodes)} nodes, "
        f"{len(local.elements)} elements"
    )
    local_area_conserved = math.isclose(local_area, coarse_area, rel_tol=1e-9)
    print(f"  Total area: {local_area:.6f} m^2 (conserved: {local_area_conserved})")
    local_validation = generate_validation_report(local)
    print(f"  Validation status: {local_validation.status} (hanging nodes are geometrically valid)")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    _plot_mesh(axes[0], coarse, "Coarse (4x2)")
    _plot_mesh(axes[1], uniform, "Uniform refinement")
    _plot_mesh(axes[2], local, "Local refinement (element 1)")
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=150)
    print(f"\nSaved comparison plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
