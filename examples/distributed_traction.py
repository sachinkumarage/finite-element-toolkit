"""Example: rectangular plate, fixed boundary, distributed traction, Version 9.

This script demonstrates the complete Version 9 workflow:

    Rectangle -> automatic Q4 mesh -> select left boundary -> fix it
        -> select right boundary -> apply distributed traction
        -> existing StaticLinearAnalysis solver -> displacement -> stress
        -> reactions

No new solver is introduced: the distributed traction is converted into
equivalent nodal loads (see :mod:`femtoolkit.analysis.distributed_load`)
and fed to the same :class:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis`
used since Version 3.
"""

from femtoolkit.analysis import DistributedLoad, LoadCase
from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, create_quad_mesh

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
    """Build, solve, and report results for a plate under distributed tension."""
    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, formulation="plane_stress"
    )
    domain = Rectangle(width=WIDTH, height=HEIGHT)
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=NX, ny=NY, material=material, thickness=THICKNESS
    )

    load_case = LoadCase(name="Uniaxial Tension", mesh=mesh)
    load_case.fix_boundary(domain.boundary("left"), ux=0.0, uy=0.0)
    load_case.add_distributed_load(DistributedLoad(domain.boundary("right"), magnitude=TRACTION))
    result = load_case.solve()

    print_summary(domain, mesh, result)


def print_summary(domain: Rectangle, mesh: Mesh, result) -> None:
    """Print a human-readable summary of the distributed-traction analysis results."""
    print("Finite Element Toolkit")
    print("Version 9 -- Distributed Traction on a Q4 Plate")
    print("=" * 40)

    print(f"\nDomain:\n    {WIDTH} m x {HEIGHT} m, nx={NX}, ny={NY}")
    print(f"\nMaterial:\n    E = {YOUNGS_MODULUS / 1e9:.0f} GPa, v = {POISSON_RATIO}")
    print(f"\nApplied traction (right boundary, normal):\n    {TRACTION:.0f} Pa")
    print("\nBoundary conditions:\n    Left boundary: fixed (ux = uy = 0)")

    right_nodes = mesh.nodes_on_boundary(domain.boundary("right"))
    print("\nRight-boundary displacements:")
    for node in right_nodes:
        ux, uy = result.node_displacement(node.id)
        print(f"    Node {node.id}: ux = {ux:.6e} m, uy = {uy:.6e} m")

    print("\nElement sigma_x (every element must match the applied traction exactly):")
    sigma_x_values = [result.element_stress(element.id)[0] for element in mesh.elements]
    print(f"    Min: {min(sigma_x_values):.6e} Pa")
    print(f"    Max: {max(sigma_x_values):.6e} Pa")

    left_nodes = mesh.nodes_on_boundary(domain.boundary("left"))
    total_rx = sum(result.node_reaction(n.id)[0] for n in left_nodes)
    total_ry = sum(result.node_reaction(n.id)[1] for n in left_nodes)
    print(f"\nTotal left-boundary reaction:\n    Rx = {total_rx:.6e} N, Ry = {total_ry:.6e} N")

    expected_force = TRACTION * HEIGHT * THICKNESS
    print(f"\nExpected total force (traction * length * thickness):\n    {expected_force:.6e} N")

    rx_ok = abs(total_rx + expected_force) < EQUILIBRIUM_TOLERANCE
    ry_ok = abs(total_ry) < EQUILIBRIUM_TOLERANCE
    print(f"\nGlobal Equilibrium:\n    {'PASS' if rx_ok and ry_ok else 'FAIL'}")

    validation_ok = all(abs(sx - TRACTION) / TRACTION < 1e-6 for sx in sigma_x_values)
    status = "PASS" if validation_ok else "FAIL"
    print(f"\nAnalytical Validation (uniform sigma_x = traction):\n    {status}")


if __name__ == "__main__":
    main()
