"""Example: solve a two-element axial bar chain with mixed materials.

Model:

    Node 1 ---- Element 1 (Steel) ---- Node 2 ---- Element 2 (Aluminum) ---- Node 3

Node 1 is fixed (u = 0); a 5000 N tensile load is applied at node 3.
This demonstrates:

  * multiple bar elements and global stiffness assembly
  * different materials and cross-sectional areas per element
  * nodal displacement results at every node
  * per-element stress, strain, and axial force
  * reaction force and global equilibrium

Since no load is applied at the shared node (node 2), statics requires
both elements to carry the same axial force, equal to the applied end
load -- even though their stress and strain differ (different E and A).
"""

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis
from femtoolkit.materials import Material
from femtoolkit.mesh import BarElement, Mesh, Node
from femtoolkit.sections import CrossSection

STEEL_AREA = 0.01  # m^2
ALUMINUM_AREA = 0.02  # m^2
ELEMENT_LENGTH = 1.0  # m, each element
APPLIED_FORCE = 5000.0  # N


def main() -> None:
    """Build, solve, and report results for a two-element mixed-material bar."""
    steel = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    aluminum = Material(name="Aluminum", density=2700.0, youngs_modulus=70e9, poissons_ratio=0.33)
    steel_section = CrossSection(area=STEEL_AREA)
    aluminum_section = CrossSection(area=ALUMINUM_AREA)

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=ELEMENT_LENGTH, y=0.0, z=0.0)
    node_3 = Node(id=3, x=2 * ELEMENT_LENGTH, y=0.0, z=0.0)

    element_1 = BarElement(
        id=1, nodes=(node_1, node_2), material=steel, cross_section=steel_section
    )
    element_2 = BarElement(
        id=2, nodes=(node_2, node_3), material=aluminum, cross_section=aluminum_section
    )

    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    mesh.add_node(node_3)
    mesh.add_element(element_1)
    mesh.add_element(element_2)

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(node_id=node_1.id, dof=0, value=0.0))
    analysis.add_load(NodalLoad(node_id=node_3.id, dof=0, value=APPLIED_FORCE))
    result = analysis.solve()

    print("Finite Element Toolkit")
    print("Version 3 -- Two-Element Bar Analysis")
    print("=" * 42)

    print("\nElements:")
    print(f"    Element 1: Steel,    A = {STEEL_AREA} m^2, L = {ELEMENT_LENGTH} m")
    print(f"    Element 2: Aluminum, A = {ALUMINUM_AREA} m^2, L = {ELEMENT_LENGTH} m")
    print(f"\nApplied load at node {node_3.id}:\n    {APPLIED_FORCE:.0f} N")

    print("\nNodal displacements:")
    for node in mesh.nodes:
        print(f"    Node {node.id}: {result.displacement(node.id):.6e} m")

    print(f"\nReaction at node {node_1.id}:\n    {result.reaction(node_1.id):.6e} N")

    print("\nElement results:")
    for element in (element_1, element_2):
        print(f"    Element {element.id} ({element.material.name}):")
        print(f"        Strain:      {result.element_strain(element.id):.6e}")
        print(f"        Stress:      {result.element_stress(element.id):.6e} Pa")
        print(f"        Axial force: {result.element_axial_force(element.id):.6e} N")

    equilibrium_residual = APPLIED_FORCE + result.reaction(node_1.id)
    equilibrium_ok = abs(equilibrium_residual) < 1e-6
    print(f"\nGlobal equilibrium:\n    {'PASS' if equilibrium_ok else 'FAIL'}")

    axial_forces_match = abs(result.element_axial_force(1) - result.element_axial_force(2)) < 1e-6
    print(f"\nUniform axial force across elements:\n    {'PASS' if axial_forces_match else 'FAIL'}")


if __name__ == "__main__":
    main()
