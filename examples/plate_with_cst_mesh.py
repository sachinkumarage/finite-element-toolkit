"""Example: rectangular plate with a CST (triangular) mesh, Version 9.

This script demonstrates that the Version 9 geometry/boundary/load
workflow works identically for a triangular mesh, not just a
quadrilateral one:

    Rectangle -> automatic CST mesh -> boundary selection
        -> distributed load -> existing StaticLinearAnalysis solver

Model: the same fixed-left, traction-right plate as
examples/distributed_traction.py, but meshed with
:func:`~femtoolkit.mesh.generator.create_triangular_mesh` instead of
:func:`~femtoolkit.mesh.generator.create_quad_mesh`.

Unlike the Q4 mesh, individual CST elements do NOT each recover the
exact applied traction: each quad cell is split into two triangles
along a diagonal, and that fixed diagonal direction biases each
triangle's constant-strain approximation away from the perfectly
uniform stress field. The *average* sigma_x across all elements still
matches the traction exactly (an exact consequence of global force
equilibrium), so that average -- not a per-element check -- is the
correct validation target for this mesh type.
"""

from femtoolkit.analysis import DistributedLoad, LoadCase
from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, create_triangular_mesh

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
THICKNESS = 0.01  # m
WIDTH = 2.0  # m
HEIGHT = 1.0  # m
NX = 4
NY = 2
TRACTION = 500e3  # Pa, normal traction on the right edge (tension)
EQUILIBRIUM_TOLERANCE = 1e-3  # N


def main() -> None:
    """Build, solve, and report results for a CST-meshed plate under distributed tension."""
    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, formulation="plane_stress"
    )
    domain = Rectangle(width=WIDTH, height=HEIGHT)
    mesh = create_triangular_mesh(
        width=WIDTH, height=HEIGHT, nx=NX, ny=NY, material=material, thickness=THICKNESS
    )

    load_case = LoadCase(name="Uniaxial Tension (CST)", mesh=mesh)
    load_case.fix_boundary(domain.boundary("left"), ux=0.0, uy=0.0)
    load_case.add_distributed_load(DistributedLoad(domain.boundary("right"), magnitude=TRACTION))
    result = load_case.solve()

    print_summary(domain, mesh, result)


def print_summary(domain: Rectangle, mesh: Mesh, result) -> None:
    """Print a human-readable summary of the CST-mesh distributed-traction analysis results."""
    print("Finite Element Toolkit")
    print("Version 9 -- Distributed Traction on a CST-Meshed Plate")
    print("=" * 40)

    print(f"\nDomain:\n    {WIDTH} m x {HEIGHT} m, nx={NX}, ny={NY}")
    print(f"\nMesh:\n    {len(mesh.nodes)} nodes, {len(mesh.elements)} CST elements")
    print(f"\nApplied traction (right boundary, normal):\n    {TRACTION:.0f} Pa")

    print("\nElement sigma_x (varies per triangle; the average must match the traction):")
    sigma_x_values = [result.element_stress(element.id)[0] for element in mesh.elements]
    average_sigma_x = sum(sigma_x_values) / len(sigma_x_values)
    print(f"    Min:     {min(sigma_x_values):.6e} Pa")
    print(f"    Max:     {max(sigma_x_values):.6e} Pa")
    print(f"    Average: {average_sigma_x:.6e} Pa")

    left_nodes = mesh.nodes_on_boundary(domain.boundary("left"))
    total_rx = sum(result.node_reaction(n.id)[0] for n in left_nodes)
    total_ry = sum(result.node_reaction(n.id)[1] for n in left_nodes)
    print(f"\nTotal left-boundary reaction:\n    Rx = {total_rx:.6e} N, Ry = {total_ry:.6e} N")

    expected_force = TRACTION * HEIGHT * THICKNESS
    print(f"\nExpected total force (traction * length * thickness):\n    {expected_force:.6e} N")

    rx_ok = abs(total_rx + expected_force) < EQUILIBRIUM_TOLERANCE
    ry_ok = abs(total_ry) < EQUILIBRIUM_TOLERANCE
    print(f"\nGlobal Equilibrium:\n    {'PASS' if rx_ok and ry_ok else 'FAIL'}")

    validation_ok = abs(average_sigma_x - TRACTION) / TRACTION < 1e-6
    status = "PASS" if validation_ok else "FAIL"
    print(f"\nAnalytical Validation (average sigma_x = traction):\n    {status}")


if __name__ == "__main__":
    main()
