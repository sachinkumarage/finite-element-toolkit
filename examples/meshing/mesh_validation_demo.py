"""Example: structured mesh validation reporting, Version 25.

Demonstrates :func:`~femtoolkit.mesh.validation.generate_validation_report`
on three meshes -- a clean generated mesh, a mesh with an isolated node,
and a mesh with a duplicated element -- printing the full
:func:`~femtoolkit.mesh.validation.format_report` text report for each,
the same non-raising, structured-report API the GUI's Mesh page (Model
Preparation -> Validate Mesh) calls.
"""

from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, Node
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.generator import create_quad_mesh
from femtoolkit.mesh.validation import format_report, generate_validation_report


def main() -> None:
    """Validate a clean mesh, a mesh with an isolated node, and one with a duplicate element."""
    material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")

    print("Finite Element Toolkit")
    print("Version 25 -- Mesh Validation Reporting")
    print("=" * 41)

    print("\n--- Clean generated mesh ---")
    clean_mesh = create_quad_mesh(
        width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01
    )
    print(format_report(generate_validation_report(clean_mesh)))

    print("\n\n--- Mesh with an isolated node ---")
    isolated_mesh = create_quad_mesh(
        width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01
    )
    isolated_mesh.add_node(Node(id=9999, x=100.0, y=100.0, z=0.0))
    print(format_report(generate_validation_report(isolated_mesh)))

    print("\n\n--- Mesh with a duplicated element ---")
    duplicate_mesh = Mesh()
    triangle_nodes = (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=1.0, y=0.0, z=0.0),
        Node(id=3, x=0.0, y=1.0, z=0.0),
    )
    for node in triangle_nodes:
        duplicate_mesh.add_node(node)
    duplicate_mesh.add_element(
        CSTElement2D(id=1, nodes=triangle_nodes, material=material, thickness=0.01)
    )
    duplicate_mesh.add_element(
        CSTElement2D(id=2, nodes=triangle_nodes, material=material, thickness=0.01)
    )
    print(format_report(generate_validation_report(duplicate_mesh)))


if __name__ == "__main__":
    main()
