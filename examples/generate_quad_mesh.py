"""Example: generate a structured Q4 mesh automatically, Version 8.

This script demonstrates the complete Version 8 mesh-generation
workflow:

    Geometry (width, height, nx, ny) -> create_quad_mesh -> Mesh
        -> node/element counts -> total area -> quality summary

instead of manually creating every :class:`~femtoolkit.mesh.node.Node`
and :class:`~femtoolkit.mesh.quad_element.QuadElement2D` by hand (see
examples/single_quad_element.py and examples/two_element_plate.py from
Version 7 for what that looks like at a small scale).
"""

from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, create_quad_mesh

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
THICKNESS = 0.01  # m
WIDTH = 2.0  # m
HEIGHT = 1.0  # m
NX = 4
NY = 2


def main() -> None:
    """Generate a rectangular Q4 mesh and report its statistics."""
    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, formulation="plane_stress"
    )

    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=NX, ny=NY, material=material, thickness=THICKNESS
    )

    print_summary(mesh)


def print_summary(mesh: Mesh) -> None:
    """Print a human-readable summary of the generated mesh."""
    print("Finite Element Toolkit")
    print("Version 8 -- Structured Q4 Mesh Generation")
    print("=" * 40)

    print(f"\nDomain:\n    {WIDTH} m x {HEIGHT} m")
    print(f"\nSubdivisions:\n    nx = {NX}, ny = {NY}")

    expected_nodes = (NX + 1) * (NY + 1)
    expected_elements = NX * NY
    print(f"\nNode count:\n    {len(mesh.nodes)} (expected {expected_nodes})")
    print(f"\nElement count:\n    {len(mesh.elements)} (expected {expected_elements})")

    total_area = sum(mesh.element_area(element.id) for element in mesh.elements)
    expected_area = WIDTH * HEIGHT
    print(f"\nTotal mesh area:\n    {total_area:.6f} m^2 (expected {expected_area:.6f} m^2)")

    summary = mesh.quality_summary()
    print("\nQuality summary:")
    print(f"    Nodes:            {summary.num_nodes}")
    print(f"    Elements:         {summary.num_elements}")
    print(f"    Min area:         {summary.min_area:.6f} m^2")
    print(f"    Max area:         {summary.max_area:.6f} m^2")
    print(f"    Min edge length:  {summary.min_edge_length:.6f} m")
    print(f"    Max edge length:  {summary.max_edge_length:.6f} m")
    print(f"    Min quality:      {summary.min_quality:.6f}")
    print(f"    Max quality:      {summary.max_quality:.6f}")
    print(f"    Average quality:  {summary.average_quality:.6f}")
    print(f"    Invalid elements: {summary.num_invalid_elements}")

    node_count_ok = len(mesh.nodes) == expected_nodes
    element_count_ok = len(mesh.elements) == expected_elements
    area_ok = abs(total_area - expected_area) / expected_area < 1e-9
    validation_ok = node_count_ok and element_count_ok and area_ok
    print(f"\nValidation:\n    {'PASS' if validation_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
