"""Example: CST patch test, demonstrating the constant-strain property, Version 6.

The defining mathematical property of the constant strain triangle: for
any linear nodal displacement field ``u = a*x + b*y``, ``v = c*x + d*y``,
a CST element must reproduce the *exact* constant strain field
``[epsilon_x, epsilon_y, gamma_xy] = [a, d, b+c]`` -- not an
approximation of it, because the element's shape functions are
themselves linear.

This script prescribes every nodal DOF directly from a chosen linear
field (so the solver has nothing left to solve for), recovers the strain
through the full analysis workflow, and compares it against the
analytically expected value. This is the same case independently
validated in tests/validation/test_cst_patch.py.
"""

from femtoolkit.analysis import BoundaryCondition, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import CSTElement2D, Mesh, Node

YOUNGS_MODULUS = 70e9  # Pa (aluminum)
POISSON_RATIO = 0.33
THICKNESS = 0.005  # m

# Linear displacement field: u = A*x + B*y, v = C*x + D*y.
A, B, C, D = 0.004, -0.002, 0.0015, 0.003
EXPECTED_STRAIN = (A, D, B + C)
TOLERANCE = 1e-10


def main() -> None:
    """Build, solve, and report the patch-test result for a single CST element."""
    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, formulation="plane_stress"
    )

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=0.4, y=1.0, z=0.0)  # an arbitrary (non-right) triangle
    element = CSTElement2D(
        id=1, nodes=(node_1, node_2, node_3), material=material, thickness=THICKNESS
    )

    mesh = Mesh()
    for node in (node_1, node_2, node_3):
        mesh.add_node(node)
    mesh.add_element(element)

    analysis = StaticLinearAnalysis(mesh)
    for node in (node_1, node_2, node_3):
        u = A * node.x + B * node.y
        v = C * node.x + D * node.y
        analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.X, u))
        analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.Y, v))
    result = analysis.solve()

    computed_strain = tuple(result.element_strain(1))
    error = max(abs(c - e) for c, e in zip(computed_strain, EXPECTED_STRAIN, strict=True))
    patch_test_ok = error < TOLERANCE

    print("CST Patch Test")
    print("=" * 40)

    print(f"\nPrescribed field:\n    u = {A}*x + ({B})*y\n    v = {C}*x + {D}*y")

    print(f"\nExpected strain:\n    epsilon_x={EXPECTED_STRAIN[0]:.6e}, "
          f"epsilon_y={EXPECTED_STRAIN[1]:.6e}, gamma_xy={EXPECTED_STRAIN[2]:.6e}")

    print(f"\nComputed strain:\n    epsilon_x={computed_strain[0]:.6e}, "
          f"epsilon_y={computed_strain[1]:.6e}, gamma_xy={computed_strain[2]:.6e}")

    print(f"\nError:\n    {error:.3e}")

    print(f"\nPatch test:\n    {'PASS' if patch_test_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
