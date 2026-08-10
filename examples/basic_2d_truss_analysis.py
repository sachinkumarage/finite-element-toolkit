"""Example: solve a small triangular 2D truss with Version 4.

This script demonstrates the complete Version 4 structural analysis
workflow:

    Material -> CrossSection -> Nodes -> TrussElements -> Mesh
        -> BoundaryConditions -> Loads -> StaticLinearAnalysis -> Results
        -> Engineering validation

Model (a simple "tent" triangle with 45-degree diagonals):

              Node 3 (1, 1)
               /\\
              /  \\
             /    \\
            /      \\
    Node 1 ---------- Node 2
     (0,0)   L=2m     (2,0)

    E = 200 GPa, A = 0.001 m^2 for every member.
    Node 1: pinned (ux = uy = 0)
    Node 2: rollered (uy = 0)
    Node 3: Fy = -1000 N (downward)

This is the same geometry validated analytically (by the method of
joints) in tests/validation/test_triangular_truss.py; the results here
are cross-checked against that same hand-derived solution.
"""

import math

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import Material
from femtoolkit.mesh import Mesh, Node, TrussElement2D
from femtoolkit.sections import CrossSection

AREA = 0.001  # m^2, every member
APPLIED_LOAD = 1000.0  # N, downward at the apex
EQUILIBRIUM_TOLERANCE = 1e-6  # N
VALIDATION_TOLERANCE = 1e-6  # relative


def main() -> None:
    """Build, solve, and report results for a triangular truss model."""
    steel = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    section = CrossSection(area=AREA)

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=1.0, y=1.0, z=0.0)

    base = TrussElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)
    left_diagonal = TrussElement2D(
        id=2, nodes=(node_1, node_3), material=steel, cross_section=section
    )
    right_diagonal = TrussElement2D(
        id=3, nodes=(node_2, node_3), material=steel, cross_section=section
    )

    mesh = Mesh()
    for node in (node_1, node_2, node_3):
        mesh.add_node(node)
    for element in (base, left_diagonal, right_diagonal):
        mesh.add_element(element)

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.X, value=0.0))
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.Y, value=0.0))
    analysis.add_boundary_condition(BoundaryCondition(node_id=2, dof=TranslationDOF.Y, value=0.0))
    analysis.add_load(NodalLoad(node_id=3, dof=TranslationDOF.Y, value=-APPLIED_LOAD))
    result = analysis.solve()

    print_analysis_summary(mesh, analysis, result)


def print_analysis_summary(mesh: Mesh, analysis: StaticLinearAnalysis, result) -> None:
    """Print a human-readable summary of the truss analysis results."""
    print("Finite Element Toolkit")
    print("Version 4 -- 2D Truss Analysis")
    print("=" * 40)

    print("\nNodes:")
    for node in mesh.nodes:
        print(f"    Node {node.id}: ({node.x}, {node.y}) m")

    print("\nElements:")
    for element in mesh.elements:
        n1, n2 = element.nodes
        print(f"    Element {element.id}: Node {n1.id} -- Node {n2.id}, L = {element.length} m")

    print(f"\nApplied Loads:\n    Node 3: Fy = {-APPLIED_LOAD:.0f} N")

    print("\nNodal Displacements:")
    for node in mesh.nodes:
        ux, uy = result.node_displacement(node.id)
        print(f"    Node {node.id}: ux = {ux:.6e} m, uy = {uy:.6e} m")

    print("\nReactions:")
    for node_id in (1, 2):
        rx, ry = result.node_reaction(node_id)
        print(f"    Node {node_id}: Rx = {rx:.6e} N, Ry = {ry:.6e} N")

    print("\nElement Results:")
    for element in mesh.elements:
        print(f"\n    Element {element.id}")
        print(f"        Axial Force: {result.element_axial_force(element.id):.6e} N")
        print(f"        Stress:      {result.element_stress(element.id):.6e} Pa")
        print(f"        Strain:      {result.element_strain(element.id):.6e}")

    rx1, ry1 = result.node_reaction(1)
    rx2, ry2 = result.node_reaction(2)
    equilibrium_ok = (
        abs(rx1 + rx2) < EQUILIBRIUM_TOLERANCE
        and abs(ry1 + ry2 - APPLIED_LOAD) < EQUILIBRIUM_TOLERANCE
    )
    print(f"\nGlobal Equilibrium:\n    {'PASS' if equilibrium_ok else 'FAIL'}")

    expected_base_force = APPLIED_LOAD / 2
    expected_diagonal_force = -APPLIED_LOAD / math.sqrt(2)
    validation_ok = (
        _relative_error(result.element_axial_force(1), expected_base_force) < VALIDATION_TOLERANCE
        and _relative_error(result.element_axial_force(2), expected_diagonal_force)
        < VALIDATION_TOLERANCE
        and _relative_error(result.element_axial_force(3), expected_diagonal_force)
        < VALIDATION_TOLERANCE
    )
    print(f"\nEngineering Validation:\n    {'PASS' if validation_ok else 'FAIL'}")


def _relative_error(computed: float, expected: float) -> float:
    """Relative error between a computed and an expected value."""
    return abs(computed - expected) / abs(expected)


if __name__ == "__main__":
    main()
