"""Example: solve a single Q4 element under uniaxial tension with Version 7.

This script demonstrates the complete Version 7 isoparametric continuum
workflow:

    Material (LinearElastic2D) -> Nodes -> QuadElement2D -> Mesh
        -> Boundary conditions -> Loads -> Analysis -> Solution
        -> Element strain -> Element stress -> Von Mises stress

Model:

    Node4 -------- Node3
     |              |
     |              |
     |              |
    Node1 -------- Node2

    Node 1 = (0, 0), Node 2 = (1, 0), Node 3 = (1, 1), Node 4 = (0, 1)
    E = 200 GPa, v = 0.3 (plane stress), t = 0.01 m
    Node 1: fixed (ux = uy = 0), Node 4: rollered (ux = 0)
    Fx = 2500 N each at Node 2 and Node 3

Behind the scenes, the element's 8x8 stiffness matrix is assembled from
four Gauss points: at each, the bilinear shape function derivatives are
evaluated in natural coordinates, transformed to physical coordinates via
the isoparametric Jacobian, assembled into a strain-displacement matrix,
and combined into the integrand B^T*D*B*det(J) -- summed with unit Gauss
weights to give Ke. The resulting stress state should closely match a
uniaxial condition: sigma_x is large, sigma_y and tau_xy are negligible.
"""

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, Node, QuadElement2D

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
THICKNESS = 0.01  # m
APPLIED_LOAD = 5000.0  # N, total, split evenly between Node 2 and Node 3
EQUILIBRIUM_TOLERANCE = 1e-6  # N


def main() -> None:
    """Build, solve, and report results for a single Q4 element model."""
    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, formulation="plane_stress"
    )

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=1.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=1.0, z=0.0)
    element = QuadElement2D(
        id=1, nodes=(node_1, node_2, node_3, node_4), material=material, thickness=THICKNESS
    )

    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4):
        mesh.add_node(node)
    mesh.add_element(element)

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.X, value=0.0))
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.Y, value=0.0))
    analysis.add_boundary_condition(BoundaryCondition(node_id=4, dof=TranslationDOF.X, value=0.0))
    analysis.add_load(NodalLoad(node_id=2, dof=TranslationDOF.X, value=APPLIED_LOAD / 2))
    analysis.add_load(NodalLoad(node_id=3, dof=TranslationDOF.X, value=APPLIED_LOAD / 2))
    result = analysis.solve()

    print_summary(mesh, result)


def print_summary(mesh: Mesh, result) -> None:
    """Print a human-readable summary of the single-Q4-element analysis results."""
    print("Finite Element Toolkit")
    print("Version 7 -- Single Q4 Element (Uniaxial Tension)")
    print("=" * 40)

    print(f"\nMaterial:\n    E = {YOUNGS_MODULUS / 1e9:.0f} GPa, v = {POISSON_RATIO}")
    print(f"\nThickness:\n    {THICKNESS} m")

    print("\nNodes:")
    for node in mesh.nodes:
        print(f"    Node {node.id}: ({node.x}, {node.y}) m")

    element = mesh.elements[0]
    print(f"\nElement area:\n    {element.area} m^2")

    print(f"\nApplied Load:\n    Node 2: Fx = {APPLIED_LOAD / 2:.0f} N")
    print(f"    Node 3: Fx = {APPLIED_LOAD / 2:.0f} N")

    print("\nNodal Displacements:")
    for node in mesh.nodes:
        ux, uy = result.node_displacement(node.id)
        print(f"    Node {node.id}: ux = {ux:.6e} m, uy = {uy:.6e} m")

    print("\nReactions:")
    for node_id in (1, 4):
        rx, ry = result.node_reaction(node_id)
        print(f"    Node {node_id}: Rx = {rx:.6e} N, Ry = {ry:.6e} N")

    sigma_x, sigma_y, tau_xy = result.element_stress(1)
    epsilon_x, epsilon_y, gamma_xy = result.element_strain(1)
    sigma_1, sigma_2 = result.element_principal_stresses(1)

    print("\nElement Strain (at element center):")
    print(
        f"    epsilon_x = {epsilon_x:.6e}, epsilon_y = {epsilon_y:.6e}, "
        f"gamma_xy = {gamma_xy:.6e}"
    )

    print("\nElement Stress (at element center):")
    print(f"    sigma_x = {sigma_x:.6e} Pa, sigma_y = {sigma_y:.6e} Pa, tau_xy = {tau_xy:.6e} Pa")

    print(f"\nVon Mises Stress:\n    {result.element_von_mises(1):.6e} Pa")
    print(f"\nPrincipal Stresses:\n    sigma_1 = {sigma_1:.6e} Pa, sigma_2 = {sigma_2:.6e} Pa")

    rx1, ry1 = result.node_reaction(1)
    rx4, ry4 = result.node_reaction(4)
    equilibrium_ok = (
        abs(rx1 + rx4 + APPLIED_LOAD) < EQUILIBRIUM_TOLERANCE
        and abs(ry1 + ry4) < EQUILIBRIUM_TOLERANCE
    )
    print(f"\nGlobal Equilibrium:\n    {'PASS' if equilibrium_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
