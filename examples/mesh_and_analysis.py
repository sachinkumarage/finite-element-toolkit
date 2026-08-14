"""Example: automatically generated mesh solved by the existing FEM solver, Version 8.

This script demonstrates the full Version 8 pipeline:

    Geometry (width, height, nx, ny)
        -> create_quad_mesh (automatic mesh generation)
        -> Boundary conditions
        -> Loads
        -> Existing StaticLinearAnalysis solver (unchanged since Version 3)
        -> Displacement
        -> Stress

No new solver is introduced: the automatically generated mesh is passed
straight into the same :class:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis`
used by every hand-built mesh in Versions 3-7.

Model: a plate fixed along its left edge (ux=0 for every node on that
edge, uy=0 additionally pinned at the bottom-left corner for stability)
and pulled by a total horizontal force, distributed evenly across the
right-edge nodes -- the discrete-nodal-load equivalent of a uniform
tensile traction.
"""

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, create_quad_mesh

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
THICKNESS = 0.01  # m
WIDTH = 2.0  # m
HEIGHT = 1.0  # m
NX = 4
NY = 2
TOTAL_LOAD = 10000.0  # N, distributed across the right-edge nodes
EQUILIBRIUM_TOLERANCE = 1e-6  # N


def main() -> None:
    """Generate a mesh, solve it, and report displacement and stress results."""
    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, formulation="plane_stress"
    )
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=NX, ny=NY, material=material, thickness=THICKNESS
    )

    left_edge_nodes = [node for node in mesh.nodes if node.x == 0.0]
    right_edge_nodes = [node for node in mesh.nodes if node.x == WIDTH]

    analysis = StaticLinearAnalysis(mesh)
    for node in left_edge_nodes:
        analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.X, 0.0))
    # Pin one node in Y to remove the remaining rigid-body (vertical
    # translation) freedom, without over-constraining the free edges.
    analysis.add_boundary_condition(BoundaryCondition(left_edge_nodes[0].id, TranslationDOF.Y, 0.0))

    load_per_node = TOTAL_LOAD / len(right_edge_nodes)
    for node in right_edge_nodes:
        analysis.add_load(NodalLoad(node.id, TranslationDOF.X, load_per_node))

    result = analysis.solve()

    print_summary(mesh, result, left_edge_nodes, right_edge_nodes)


def print_summary(mesh: Mesh, result, left_edge_nodes, right_edge_nodes) -> None:
    """Print a human-readable summary of the generated-mesh analysis results."""
    print("Finite Element Toolkit")
    print("Version 8 -- Generated Mesh + Existing FEM Solver")
    print("=" * 40)

    print(f"\nDomain:\n    {WIDTH} m x {HEIGHT} m, nx={NX}, ny={NY}")
    print(f"\nMesh:\n    {len(mesh.nodes)} nodes, {len(mesh.elements)} Q4 elements")
    print(
        f"\nApplied load:\n    {TOTAL_LOAD:.0f} N, split across "
        f"{len(right_edge_nodes)} right-edge nodes"
    )

    print("\nRight-edge displacements:")
    for node in right_edge_nodes:
        ux, uy = result.node_displacement(node.id)
        print(f"    Node {node.id}: ux = {ux:.6e} m, uy = {uy:.6e} m")

    reaction_fx_total = sum(result.node_reaction(node.id)[0] for node in left_edge_nodes)
    reaction_fy_total = sum(result.node_reaction(node.id)[1] for node in left_edge_nodes)
    print(
        f"\nTotal left-edge reaction:\n    Rx = {reaction_fx_total:.6e} N, "
        f"Ry = {reaction_fy_total:.6e} N"
    )

    print("\nElement sigma_x (should cluster near the applied uniaxial stress):")
    sigma_x_values = []
    for element in mesh.elements:
        sigma_x, _, _ = result.element_stress(element.id)
        sigma_x_values.append(sigma_x)
    average_sigma_x = sum(sigma_x_values) / len(sigma_x_values)
    print(f"    Average sigma_x: {average_sigma_x:.6e} Pa")
    print(f"    Min sigma_x:     {min(sigma_x_values):.6e} Pa")
    print(f"    Max sigma_x:     {max(sigma_x_values):.6e} Pa")

    expected_sigma_x = TOTAL_LOAD / (HEIGHT * THICKNESS)
    print(f"\nExpected uniform sigma_x (P / (H*t)):\n    {expected_sigma_x:.6e} Pa")

    equilibrium_ok = (
        abs(reaction_fx_total + TOTAL_LOAD) < EQUILIBRIUM_TOLERANCE
        and abs(reaction_fy_total) < EQUILIBRIUM_TOLERANCE
    )
    print(f"\nGlobal Equilibrium:\n    {'PASS' if equilibrium_ok else 'FAIL'}")

    validation_ok = abs(average_sigma_x - expected_sigma_x) / expected_sigma_x < 1e-6
    print(f"\nAnalytical Validation (average sigma_x):\n    {'PASS' if validation_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
