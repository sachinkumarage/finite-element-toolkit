"""Example: a triangular truss combining multiple materials and orientations.

Model:

              Node 3 (2, 3)
               /\\
              /  \\
             /    \\
            /      \\
    Node 1 ---------- Node 2
     (0,0)   L=4m     (4,0)

    Element 1 (base, horizontal):       Steel,    A = 0.002 m^2
    Element 2 (left diagonal, ~56 deg): Aluminum, A = 0.001 m^2
    Element 3 (right diagonal, ~124 deg): Aluminum, A = 0.0012 m^2

    Node 1: pinned (ux = uy = 0)
    Node 2: rollered (uy = 0)
    Node 3: Fx = 2000 N, Fy = -3000 N

This demonstrates multiple materials, multiple cross-sectional areas, and
multiple element orientations within a single structural analysis.
"""

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import Material
from femtoolkit.mesh import Mesh, Node, TrussElement2D
from femtoolkit.sections import CrossSection


def main() -> None:
    """Build, solve, and report results for a multi-material truss model."""
    steel = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    aluminum = Material(name="Aluminum", density=2700.0, youngs_modulus=70e9, poissons_ratio=0.33)

    base_section = CrossSection(area=0.002)
    left_section = CrossSection(area=0.001)
    right_section = CrossSection(area=0.0012)

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=4.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=2.0, y=3.0, z=0.0)

    base = TrussElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=base_section)
    left_diagonal = TrussElement2D(
        id=2, nodes=(node_1, node_3), material=aluminum, cross_section=left_section
    )
    right_diagonal = TrussElement2D(
        id=3, nodes=(node_2, node_3), material=aluminum, cross_section=right_section
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
    analysis.add_load(NodalLoad(node_id=3, dof=TranslationDOF.X, value=2000.0))
    analysis.add_load(NodalLoad(node_id=3, dof=TranslationDOF.Y, value=-3000.0))
    result = analysis.solve()

    print("Finite Element Toolkit")
    print("Version 4 -- Multi-Material Truss Analysis")
    print("=" * 44)

    print("\nElements:")
    for element in mesh.elements:
        c, s = element.direction_cosines
        print(
            f"    Element {element.id} ({element.material.name}, "
            f"A={element.cross_section.area} m^2): "
            f"L={element.length:.3f} m, cos={c:.3f}, sin={s:.3f}"
        )

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
        print(f"\n    Element {element.id} ({element.material.name})")
        print(f"        Axial Force: {result.element_axial_force(element.id):.6e} N")
        print(f"        Stress:      {result.element_stress(element.id):.6e} Pa")
        print(f"        Strain:      {result.element_strain(element.id):.6e}")

    rx1, ry1 = result.node_reaction(1)
    rx2, ry2 = result.node_reaction(2)
    sum_fx = rx1 + rx2 + 2000.0
    sum_fy = ry1 + ry2 - 3000.0
    equilibrium_ok = abs(sum_fx) < 1e-6 and abs(sum_fy) < 1e-6
    print(f"\nGlobal Equilibrium:\n    {'PASS' if equilibrium_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
