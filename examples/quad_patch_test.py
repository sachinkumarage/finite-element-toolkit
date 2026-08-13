"""Example: Q4 patch test, demonstrating exact linear-field reproduction, Version 7.

A linear displacement field, ``u = a*x + b*y``, ``v = c*x + d*y``, is a
special case of the Q4 element's bilinear interpolation, so a Q4 element
must reproduce the *exact* constant strain field
``[epsilon_x, epsilon_y, gamma_xy] = [a, d, b+c]`` -- not an approximation
of it -- at every point within the element, despite its shape functions
being bilinear (not linear) and its strain-displacement matrix varying
from Gauss point to Gauss point.

This script prescribes every nodal DOF directly from a chosen linear
field (so the solver has nothing left to solve for), recovers the strain
through the full analysis workflow, and compares it against the
analytically expected value. This is the same case independently
validated in tests/validation/test_quad_patch.py.
"""

from femtoolkit.analysis import BoundaryCondition, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, Node, QuadElement2D

YOUNGS_MODULUS = 70e9  # Pa (aluminum)
POISSON_RATIO = 0.33
THICKNESS = 0.005  # m

# Linear displacement field: u = A*x + B*y, v = C*x + D*y.
A, B, C, D = 0.004, -0.002, 0.0015, 0.003
EXPECTED_STRAIN = (A, D, B + C)
TOLERANCE = 1e-9


def main() -> None:
    """Build, solve, and report the patch-test result for a single Q4 element."""
    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, formulation="plane_stress"
    )

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=2.5, y=1.5, z=0.0)  # a general (non-rectangular) quad
    node_4 = Node(id=4, x=0.3, y=1.2, z=0.0)
    element = QuadElement2D(
        id=1, nodes=(node_1, node_2, node_3, node_4), material=material, thickness=THICKNESS
    )

    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4):
        mesh.add_node(node)
    mesh.add_element(element)

    analysis = StaticLinearAnalysis(mesh)
    for node in (node_1, node_2, node_3, node_4):
        u = A * node.x + B * node.y
        v = C * node.x + D * node.y
        analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.X, u))
        analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.Y, v))
    result = analysis.solve()

    computed_strain = tuple(result.element_strain(1))
    error = max(abs(c - e) for c, e in zip(computed_strain, EXPECTED_STRAIN, strict=True))
    patch_test_ok = error < TOLERANCE

    print("Q4 Patch Test")
    print("=" * 40)

    print(f"\nPrescribed field:\n    u = {A}*x + ({B})*y\n    v = {C}*x + {D}*y")

    print(
        f"\nExpected strain:\n    epsilon_x={EXPECTED_STRAIN[0]:.6e}, "
        f"epsilon_y={EXPECTED_STRAIN[1]:.6e}, gamma_xy={EXPECTED_STRAIN[2]:.6e}"
    )

    print(
        f"\nComputed strain:\n    epsilon_x={computed_strain[0]:.6e}, "
        f"epsilon_y={computed_strain[1]:.6e}, gamma_xy={computed_strain[2]:.6e}"
    )

    print(f"\nError:\n    {error:.3e}")

    print(f"\nPatch test:\n    {'PASS' if patch_test_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
