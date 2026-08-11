"""Example: solve a simple portal frame under lateral load with Version 5.

This script demonstrates a complete Version 5 structural analysis with
multiple connected frame elements of different orientations (two vertical
columns and one horizontal beam) sharing a structure:

    Node 3 -------- beam -------- Node 4
      |                             |
    column 1                    column 2
      |                             |
    Node 1 (fixed)              Node 2 (fixed)

    Node 1 = (0, 0), Node 2 = (4, 0), Node 3 = (0, 3), Node 4 = (4, 3)
    E = 200 GPa, A = 0.01 m^2, I = 8.333e-6 m^4
    Fx = 2000 N applied at Node 3 (lateral/wind load)
    Node 1, Node 2: fixed (ux = uy = rz = 0)

This is the same model independently validated (via global equilibrium
and mechanics-derived consistency checks, since a fixed-fixed portal
frame under an asymmetric lateral load is statically indeterminate) in
tests/validation/test_portal_frame.py.
"""

from femtoolkit.analysis import (
    BoundaryCondition,
    NodalLoad,
    RotationDOF,
    StaticLinearAnalysis,
    TranslationDOF,
)
from femtoolkit.materials import Material
from femtoolkit.mesh import FrameElement2D, Mesh, Node
from femtoolkit.sections import CrossSection

AREA = 0.01  # m^2
SECOND_MOMENT_OF_AREA = 8.333e-6  # m^4
WIDTH = 4.0  # m
HEIGHT = 3.0  # m
LATERAL_LOAD = 2000.0  # N
EQUILIBRIUM_TOLERANCE = 1e-3  # N and N*m


def main() -> None:
    """Build, solve, and report results for a portal frame model."""
    steel = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    section = CrossSection(area=AREA, second_moment_of_area=SECOND_MOMENT_OF_AREA)

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=WIDTH, y=0.0, z=0.0)
    node_3 = Node(id=3, x=0.0, y=HEIGHT, z=0.0)
    node_4 = Node(id=4, x=WIDTH, y=HEIGHT, z=0.0)

    column_1 = FrameElement2D(id=1, nodes=(node_1, node_3), material=steel, cross_section=section)
    column_2 = FrameElement2D(id=2, nodes=(node_2, node_4), material=steel, cross_section=section)
    beam = FrameElement2D(id=3, nodes=(node_3, node_4), material=steel, cross_section=section)

    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4):
        mesh.add_node(node)
    for element in (column_1, column_2, beam):
        mesh.add_element(element)

    analysis = StaticLinearAnalysis(mesh)
    for node_id in (1, 2):
        for dof in (TranslationDOF.X, TranslationDOF.Y, RotationDOF.RZ):
            analysis.add_boundary_condition(BoundaryCondition(node_id=node_id, dof=dof, value=0.0))
    analysis.add_load(NodalLoad(node_id=3, dof=TranslationDOF.X, value=LATERAL_LOAD))
    result = analysis.solve()

    print_analysis_summary(mesh, result)


def print_analysis_summary(mesh: Mesh, result) -> None:
    """Print a human-readable summary of the portal frame analysis results."""
    print("Finite Element Toolkit")
    print("Version 5 -- Portal Frame Analysis")
    print("=" * 40)

    print("\nNodes:")
    for node in mesh.nodes:
        print(f"    Node {node.id}: ({node.x}, {node.y}) m")

    print("\nElements:")
    for element in mesh.elements:
        n1, n2 = element.nodes
        print(f"    Element {element.id}: Node {n1.id} -- Node {n2.id}, L = {element.length:.3f} m")

    print(f"\nApplied Loads:\n    Node 3: Fx = {LATERAL_LOAD:.0f} N")

    print("\nNodal Displacements:")
    for node in mesh.nodes:
        ux, uy, rz = result.node_displacement(node.id)
        print(f"    Node {node.id}: ux = {ux:.6e} m, uy = {uy:.6e} m, rz = {rz:.6e} rad")

    print("\nReactions:")
    for node_id in (1, 2):
        rx, ry, mz = result.node_reaction(node_id)
        print(f"    Node {node_id}: Rx = {rx:.6e} N, Ry = {ry:.6e} N, Mz = {mz:.6e} N*m")

    print("\nElement End Forces:")
    for element in mesh.elements:
        forces = result.element_end_forces(element.id)
        print(f"\n    Element {element.id}")
        print(
            f"        Node 1: N = {forces.node_1.axial_force:.6e} N, "
            f"V = {forces.node_1.shear_force:.6e} N, M = {forces.node_1.bending_moment:.6e} N*m"
        )
        print(
            f"        Node 2: N = {forces.node_2.axial_force:.6e} N, "
            f"V = {forces.node_2.shear_force:.6e} N, M = {forces.node_2.bending_moment:.6e} N*m"
        )

    rx1, ry1, mz1 = result.node_reaction(1)
    rx2, ry2, mz2 = result.node_reaction(2)
    sum_fx = rx1 + rx2 + LATERAL_LOAD
    sum_fy = ry1 + ry2
    moment_from_reaction_2 = WIDTH * ry2
    moment_from_load = -HEIGHT * LATERAL_LOAD
    sum_mz = mz1 + mz2 + moment_from_reaction_2 + moment_from_load

    equilibrium_ok = (
        abs(sum_fx) < EQUILIBRIUM_TOLERANCE
        and abs(sum_fy) < EQUILIBRIUM_TOLERANCE
        and abs(sum_mz) < EQUILIBRIUM_TOLERANCE
    )
    print(f"\nGlobal Force Equilibrium (sum Fx):\n    {sum_fx:.6e} N")
    print(f"\nGlobal Force Equilibrium (sum Fy):\n    {sum_fy:.6e} N")
    print(f"\nGlobal Moment Equilibrium about Node 1 (sum Mz):\n    {sum_mz:.6e} N*m")
    print(f"\nGlobal Equilibrium:\n    {'PASS' if equilibrium_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
