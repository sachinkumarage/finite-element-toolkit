"""Example: geometry-aware boundary node selection, Version 9.

This script demonstrates the Version 9 geometry foundation on its own,
without running an analysis:

    Rectangle geometry -> named boundaries -> automatic mesh -> matching
    mesh nodes selected by physical location, not manually listed IDs.

Compare this to the manual approach every prior version needed --
inspecting node coordinates by hand (e.g. ``[n for n in mesh.nodes if
n.x == 0.0]``) to figure out which nodes lie on a given edge.
"""

from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, create_quad_mesh

WIDTH = 2.0  # m
HEIGHT = 1.0  # m
NX = 4
NY = 2


def main() -> None:
    """Generate a mesh and report the nodes found on each named boundary."""
    material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    domain = Rectangle(width=WIDTH, height=HEIGHT)
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=NX, ny=NY, material=material, thickness=0.01
    )

    print_summary(domain, mesh)


def print_summary(domain: Rectangle, mesh: Mesh) -> None:
    """Print a human-readable summary of the domain's named boundaries and their nodes."""
    print("Finite Element Toolkit")
    print("Version 9 -- Geometry and Boundary Selection")
    print("=" * 40)

    print(f"\nDomain:\n    {domain.width} m x {domain.height} m")
    print("\nCorners (counter-clockwise from bottom-left):")
    for corner in domain.corners:
        print(f"    ({corner.x}, {corner.y})")

    print(f"\nMesh:\n    {len(mesh.nodes)} nodes, {len(mesh.elements)} Q4 elements")

    for name in ("left", "right", "top", "bottom"):
        boundary = domain.boundary(name)
        nodes = mesh.nodes_on_boundary(boundary)
        node_ids = [node.id for node in nodes]
        print(f"\nBoundary '{name}':")
        print(f"    Outward normal: {boundary.outward_normal}")
        print(f"    Node count:     {len(nodes)}")
        print(f"    Node IDs:       {node_ids}")

    left_ids = {n.id for n in mesh.nodes_on_boundary(domain.boundary("left"))}
    bottom_ids = {n.id for n in mesh.nodes_on_boundary(domain.boundary("bottom"))}
    shared_corner = left_ids & bottom_ids
    print(f"\nBottom-left corner node (on both 'left' and 'bottom'):\n    {shared_corner}")


if __name__ == "__main__":
    main()
