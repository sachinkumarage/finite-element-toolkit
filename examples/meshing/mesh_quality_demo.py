"""Example: mesh shape-quality evaluation, Version 25.

Builds two meshes -- a well-shaped structured Q4 mesh and a mesh
containing one deliberately elongated element -- and evaluates both
through :class:`~femtoolkit.mesh.quality.evaluator.QualityEvaluator`,
demonstrating the same ``evaluator.evaluate(mesh)`` API the GUI's Mesh
page (Model Preparation -> Evaluate Quality) calls. If PyVista (the
``viz3d`` extra) is installed, also renders a 3D screenshot of the
distorted mesh colored by quality, reusing Version 23's own
visualization machinery (see
:func:`femtoolkit.gui.visualization.render_mesh_quality_screenshot`).
"""

from pathlib import Path

from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, Node, QuadElement2D
from femtoolkit.mesh.generator import create_quad_mesh
from femtoolkit.mesh.quality import QualityEvaluator

OUTPUT_PATH = Path(__file__).parent / "mesh_quality_demo.png"


def _build_distorted_mesh(material: LinearElastic2D) -> Mesh:
    """Build a two-element mesh: one square element, one badly elongated one."""
    mesh = Mesh()
    square_nodes = (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=1.0, y=0.0, z=0.0),
        Node(id=3, x=1.0, y=1.0, z=0.0),
        Node(id=4, x=0.0, y=1.0, z=0.0),
    )
    elongated_nodes = (
        Node(id=5, x=1.0, y=0.0, z=0.0),
        Node(id=6, x=9.0, y=0.0, z=0.0),
        Node(id=7, x=9.0, y=1.0, z=0.0),
        Node(id=8, x=1.0, y=1.0, z=0.0),
    )
    for node in square_nodes:
        mesh.add_node(node)
    mesh.add_element(QuadElement2D(id=1, nodes=square_nodes, material=material, thickness=0.01))
    for node in elongated_nodes[1:]:
        mesh.add_node(node)
    mesh.add_element(
        QuadElement2D(
            id=2,
            nodes=(square_nodes[1], *elongated_nodes[1:]),
            material=material,
            thickness=0.01,
        )
    )
    return mesh


def main() -> None:
    """Evaluate shape quality for a well-shaped mesh and a distorted one."""
    material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    evaluator = QualityEvaluator(poor_quality_threshold=0.3)

    print("Finite Element Toolkit")
    print("Version 25 -- Mesh Quality Evaluation")
    print("=" * 40)

    good_mesh = create_quad_mesh(
        width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01
    )
    good_report = evaluator.evaluate(good_mesh)
    print("\nWell-shaped structured mesh (4x2 Q4 grid):")
    print(f"  Elements evaluated: {good_report.num_elements_evaluated}")
    print(f"  Minimum quality:    {good_report.minimum_quality:.3f}")
    print(f"  Mean quality:       {good_report.mean_quality:.3f}")
    print(f"  Poor elements:      {good_report.poor_quality_element_ids}")

    distorted_mesh = _build_distorted_mesh(material)
    distorted_report = evaluator.evaluate(distorted_mesh)
    print("\nMesh with one badly elongated element:")
    print(f"  Elements evaluated: {distorted_report.num_elements_evaluated}")
    print(f"  Minimum quality:    {distorted_report.minimum_quality:.3f}")
    print(f"  Mean quality:       {distorted_report.mean_quality:.3f}")
    print(f"  Poor elements:      {distorted_report.poor_quality_element_ids}")
    for warning in distorted_report.warnings:
        print(f"  Warning: {warning}")

    element_2_quality = distorted_report.element_qualities[2]
    print(
        f"\nElement 2 detail: aspect_ratio={element_2_quality.aspect_ratio:.2f}, "
        f"skewness={element_2_quality.skewness:.3f}"
    )

    from femtoolkit.gui.visualization import is_pyvista_available, render_mesh_quality_screenshot

    if is_pyvista_available():
        render_mesh_quality_screenshot(
            distorted_mesh, distorted_report, OUTPUT_PATH, metric="quality"
        )
        print(f"\nSaved 3D quality screenshot to {OUTPUT_PATH}")
    else:
        print(
            "\n3D quality screenshot skipped: install PyVista with "
            'pip install "femtoolkit[viz3d]"'
        )


if __name__ == "__main__":
    main()
