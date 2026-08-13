"""Example: solve a rectangular plate meshed with two Q4 elements, Version 7.

This script demonstrates a multi-element Version 7 continuum analysis:

    Node6 ---- Node5 ---- Node4
      |    left  |  right   |
      |          |          |
    Node1 ---- Node2 ---- Node3

    Node 1=(0,0), 2=(1,0), 3=(2,0), 4=(2,1), 5=(1,1), 6=(0,1)
    "left" element: (1, 2, 5, 6); "right" element: (2, 3, 4, 5)
    E = 200 GPa, v = 0.3 (plane stress), t = 0.01 m
    Node 1: fixed (ux = uy = 0), Node 6: rollered (ux = 0)
    Fx = 500 N each at Node 3 and Node 4 -- statically equivalent to a
    uniform 100 kPa tensile traction on the right edge.

This is the same model independently validated in
tests/validation/test_two_quad_plate.py: for this particular loading, the
exact elasticity solution is itself constant stress, so both quadrilaterals
should recover the same uniform uniaxial stress state, and the shared edge
(Node 2, Node 5) demonstrates displacement continuity across the assembly.
"""

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, Node, QuadElement2D

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
THICKNESS = 0.01  # m
HEIGHT = 1.0  # m
APPLIED_LOAD = 1000.0  # N, total, split evenly between Node 3 and Node 4
EQUILIBRIUM_TOLERANCE = 1e-6  # N


def main() -> None:
    """Build, solve, and report results for a two-Q4-element plate model."""
    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, formulation="plane_stress"
    )

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=2.0, y=0.0, z=0.0)
    node_4 = Node(id=4, x=2.0, y=HEIGHT, z=0.0)
    node_5 = Node(id=5, x=1.0, y=HEIGHT, z=0.0)
    node_6 = Node(id=6, x=0.0, y=HEIGHT, z=0.0)

    left = QuadElement2D(
        id=1, nodes=(node_1, node_2, node_5, node_6), material=material, thickness=THICKNESS
    )
    right = QuadElement2D(
        id=2, nodes=(node_2, node_3, node_4, node_5), material=material, thickness=THICKNESS
    )

    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4, node_5, node_6):
        mesh.add_node(node)
    for element in (left, right):
        mesh.add_element(element)

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.X, value=0.0))
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.Y, value=0.0))
    analysis.add_boundary_condition(BoundaryCondition(node_id=6, dof=TranslationDOF.X, value=0.0))
    analysis.add_load(NodalLoad(node_id=3, dof=TranslationDOF.X, value=APPLIED_LOAD / 2))
    analysis.add_load(NodalLoad(node_id=4, dof=TranslationDOF.X, value=APPLIED_LOAD / 2))
    result = analysis.solve()

    print_summary(mesh, result)


def print_summary(mesh: Mesh, result) -> None:
    """Print a human-readable summary of the two-Q4-element plate analysis results."""
    print("Finite Element Toolkit")
    print("Version 7 -- Two-Element Q4 Plate Analysis")
    print("=" * 40)

    print(f"\nMaterial:\n    E = {YOUNGS_MODULUS / 1e9:.0f} GPa, v = {POISSON_RATIO}")
    print(f"\nThickness:\n    {THICKNESS} m")

    print("\nNodes:")
    for node in mesh.nodes:
        print(f"    Node {node.id}: ({node.x}, {node.y}) m")

    print(f"\nApplied Loads:\n    Node 3: Fx = {APPLIED_LOAD / 2:.0f} N")
    print(f"    Node 4: Fx = {APPLIED_LOAD / 2:.0f} N")

    print("\nNodal Displacements:")
    for node in mesh.nodes:
        ux, uy = result.node_displacement(node.id)
        print(f"    Node {node.id}: ux = {ux:.6e} m, uy = {uy:.6e} m")

    print("\nReactions:")
    for node_id in (1, 6):
        rx, ry = result.node_reaction(node_id)
        print(f"    Node {node_id}: Rx = {rx:.6e} N, Ry = {ry:.6e} N")

    print("\nElement Results:")
    for element in mesh.elements:
        sigma_x, sigma_y, tau_xy = result.element_stress(element.id)
        von_mises = result.element_von_mises(element.id)
        print(f"\n    Element {element.id}")
        print(f"        Area:        {element.area:.6f} m^2")
        print(
            f"        Stress:      sigma_x = {sigma_x:.6e} Pa, "
            f"sigma_y = {sigma_y:.6e} Pa, tau_xy = {tau_xy:.6e} Pa"
        )
        print(f"        Von Mises:   {von_mises:.6e} Pa")

    expected_sigma_x = APPLIED_LOAD / (HEIGHT * THICKNESS)
    print(f"\nExpected uniform sigma_x (P / (H*t)):\n    {expected_sigma_x:.6e} Pa")

    rx1, ry1 = result.node_reaction(1)
    rx6, ry6 = result.node_reaction(6)
    equilibrium_ok = (
        abs(rx1 + rx6 + APPLIED_LOAD) < EQUILIBRIUM_TOLERANCE
        and abs(ry1 + ry6) < EQUILIBRIUM_TOLERANCE
    )
    print(f"\nGlobal Equilibrium:\n    {'PASS' if equilibrium_ok else 'FAIL'}")

    sigma_x_left = result.element_stress(1)[0]
    sigma_x_right = result.element_stress(2)[0]
    validation_ok = (
        abs(sigma_x_left - expected_sigma_x) / expected_sigma_x < 1e-6
        and abs(sigma_x_right - expected_sigma_x) / expected_sigma_x < 1e-6
    )
    print(f"\nAnalytical Validation:\n    {'PASS' if validation_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
