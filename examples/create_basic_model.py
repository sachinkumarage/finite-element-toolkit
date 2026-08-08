"""Example: build a minimal finite element model with the Version 1 foundation.

This script demonstrates the domain objects available in Version 1 of the
Finite Element Toolkit: Material, Node, Element, and Mesh. It builds the
simplest possible model, a single element connecting two nodes:

    Node 1 -------- Node 2
           Element 1

No finite element analysis is performed. Version 1 provides only the data
model that future versions will build a solver on top of.
"""

from femtoolkit.materials import Material
from femtoolkit.mesh import Element, Mesh, Node


def main() -> None:
    """Build and print a summary of a minimal two-node, one-element model."""
    steel = Material(
        name="Steel",
        density=7850.0,  # kg/m^3
        youngs_modulus=200e9,  # Pa
        poissons_ratio=0.3,
    )

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)  # meters
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)  # meters

    element = Element(id=1, nodes=[node_1, node_2], material=steel)

    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    mesh.add_element(element)

    print_model_summary(mesh, steel)


def print_model_summary(mesh: Mesh, material: Material) -> None:
    """Print a human-readable summary of a mesh and its material.

    Args:
        mesh: The mesh to summarize.
        material: The material used by the model, shown for reference.
    """
    print("Finite Element Toolkit - Basic Model Summary")
    print("=" * 46)

    print(f"\nMaterial: {material.name}")
    print(f"  Density:          {material.density:,.1f} kg/m^3")
    print(f"  Young's modulus:  {material.youngs_modulus:,.1e} Pa")
    print(f"  Poisson's ratio:  {material.poissons_ratio}")

    print(f"\nNodes ({len(mesh.nodes)}):")
    for node in mesh.nodes:
        print(f"  Node {node.id}: (x={node.x}, y={node.y}, z={node.z})")

    print(f"\nElements ({len(mesh.elements)}):")
    for element in mesh.elements:
        node_ids = [node.id for node in element.nodes]
        print(f"  Element {element.id}: nodes={node_ids}, material={element.material.name}")


if __name__ == "__main__":
    main()
