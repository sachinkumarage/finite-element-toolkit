"""Example: tying two independently meshed regions with a multi-point constraint, Version 10.

Demonstrates :class:`~femtoolkit.analysis.multi_point_constraint.MultiPointConstraint`:
two bar chains that are topologically independent (they do not share any
node) but occupy the same physical location at their meeting point are
tied together with ``ux(node_2) = ux(node_3)``. If the tie holds, the two
chains behave as a single continuous bar of the combined length -- a
scenario that comes up whenever two independently generated meshes need
to be connected without re-meshing them as one (e.g. two parts modeled
by different tools or teams).

Enforcement uses the penalty method (see
:mod:`femtoolkit.analysis.multi_point_constraint`): an approximate but
standard technique, verified here against the exact analytical solution
for a single bar of the combined length.
"""

from femtoolkit.analysis import (
    BoundaryCondition,
    MultiPointConstraint,
    NodalLoad,
    StaticLinearAnalysis,
    TranslationDOF,
)
from femtoolkit.materials import Material
from femtoolkit.mesh import BarElement, Mesh, Node
from femtoolkit.sections import CrossSection

YOUNGS_MODULUS = 200e9  # Pa
AREA = 0.01  # m^2
LENGTH_1 = 1.0  # m
LENGTH_2 = 1.0  # m
FORCE = 10_000.0  # N


def main() -> None:
    """Build two tied bar chains, solve, and compare against the single-bar analytical solution."""
    steel = Material(
        name="Steel", density=7850.0, youngs_modulus=YOUNGS_MODULUS, poissons_ratio=0.3
    )
    section = CrossSection(area=AREA)

    mesh = Mesh()
    mesh.add_node(Node(1, 0.0, 0.0, 0.0))
    mesh.add_node(Node(2, LENGTH_1, 0.0, 0.0))
    mesh.add_node(Node(3, LENGTH_1, 0.0, 0.0))  # physically coincides with node 2
    mesh.add_node(Node(4, LENGTH_1 + LENGTH_2, 0.0, 0.0))
    mesh.add_element(
        BarElement(id=1, nodes=(mesh.get_node(1), mesh.get_node(2)),
                    material=steel, cross_section=section)
    )
    mesh.add_element(
        BarElement(id=2, nodes=(mesh.get_node(3), mesh.get_node(4)),
                    material=steel, cross_section=section)
    )

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(1, TranslationDOF.X, 0.0))
    analysis.add_load(NodalLoad(4, TranslationDOF.X, FORCE))
    analysis.add_multi_point_constraint(
        MultiPointConstraint(node_id_a=2, node_id_b=3, dof=TranslationDOF.X)
    )
    result = analysis.solve()

    print_summary(result)


def print_summary(result) -> None:
    """Print tied-node and tip displacements against the analytical single-bar solution."""
    print("Finite Element Toolkit")
    print("Version 10 -- Multi-Point Constraint (Tied Bar Chains)")
    print("=" * 40)

    ux_2 = result.displacement(2, TranslationDOF.X)
    ux_3 = result.displacement(3, TranslationDOF.X)
    ux_4 = result.displacement(4, TranslationDOF.X)
    reaction_1 = result.reaction(1, TranslationDOF.X)

    print(f"\nTied interface: node 2 ux = {ux_2:.6e} m, node 3 ux = {ux_3:.6e} m")
    print("    (must match closely -- penalty method, not bit-exact)")

    analytical_mid = FORCE * LENGTH_1 / (YOUNGS_MODULUS * AREA)
    analytical_tip = FORCE * (LENGTH_1 + LENGTH_2) / (YOUNGS_MODULUS * AREA)
    print("\nTip displacement (node 4):")
    print(f"    FEA: {ux_4:.6e} m, Analytical: {analytical_tip:.6e} m")
    print("Interface displacement:")
    print(f"    FEA: {ux_2:.6e} m, Analytical: {analytical_mid:.6e} m")

    print(f"\nFixed-end reaction:\n    {reaction_1:.3f} N, Expected: {-FORCE:.3f} N")

    tip_ok = abs(ux_4 - analytical_tip) / analytical_tip < 1e-4
    tie_ok = abs(ux_2 - ux_3) / analytical_mid < 1e-4
    print(f"\nValidation:\n    {'PASS' if tip_ok and tie_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
