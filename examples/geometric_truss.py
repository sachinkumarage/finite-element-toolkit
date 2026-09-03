"""Example: a large-displacement (shallow) truss, Version 16.

Demonstrates the whole point of geometric nonlinearity: a **shallow** truss
(two members meeting at a low apex, spanning a much larger horizontal
distance) under a vertical apex load produces a *large rotation* of both
members even though nodal displacements stay geometrically modest -- the
regime where the small-strain linear stiffness matrix becomes a poor
approximation. This example solves the same structure both ways and
compares them directly.
"""

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.materials import Material, SaintVenantKirchhoff1D
from femtoolkit.mesh import Mesh, Node, TrussElement2D
from femtoolkit.sections import CrossSection

X = TranslationDOF.X
Y = TranslationDOF.Y

RISE = 0.2  # m, apex height above the supports
SPAN = 5.0  # m, horizontal distance between supports
YOUNGS_MODULUS = 200e9  # Pa
AREA = 0.0005  # m^2
APEX_LOAD = -2.0e4  # N, downward


def main() -> None:
    """Build a shallow truss, solve it both linearly and geometrically nonlinearly, and compare."""
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=SPAN / 2, y=RISE, z=0.0)  # apex
    node_3 = Node(id=3, x=SPAN, y=0.0, z=0.0)
    placeholder = Material(
        name="steel", density=7850.0, youngs_modulus=YOUNGS_MODULUS, poissons_ratio=0.3
    )
    section = CrossSection(area=AREA)
    truss_1 = TrussElement2D(
        id=1, nodes=(node_1, node_2), material=placeholder, cross_section=section
    )
    truss_2 = TrussElement2D(
        id=2, nodes=(node_2, node_3), material=placeholder, cross_section=section
    )

    mesh = Mesh()
    for node in (node_1, node_2, node_3):
        mesh.add_node(node)
    mesh.add_element(truss_1)
    mesh.add_element(truss_2)

    # --- Linear (small-displacement) solution ---
    linear_analysis = StaticLinearAnalysis(mesh)
    linear_analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    linear_analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    linear_analysis.add_boundary_condition(BoundaryCondition(3, X, 0.0))
    linear_analysis.add_boundary_condition(BoundaryCondition(3, Y, 0.0))
    linear_analysis.add_load(NodalLoad(2, Y, APEX_LOAD))
    linear_result = linear_analysis.solve()

    # --- Geometrically nonlinear (large-displacement) solution ---
    material = SaintVenantKirchhoff1D(youngs_modulus=YOUNGS_MODULUS)
    settings = NonlinearSolverSettings(load_steps=20, tolerance=1e-9, max_iterations=60)
    nonlinear_analysis = NonlinearAnalysis(
        mesh, {truss_1.id: material, truss_2.id: material}, settings, geometric_nonlinearity=True
    )
    nonlinear_analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    nonlinear_analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    nonlinear_analysis.add_boundary_condition(BoundaryCondition(3, X, 0.0))
    nonlinear_analysis.add_boundary_condition(BoundaryCondition(3, Y, 0.0))
    nonlinear_analysis.add_load(NodalLoad(2, Y, APEX_LOAD))
    nonlinear_result = nonlinear_analysis.solve()

    print_summary(linear_result, nonlinear_result)


def print_summary(linear_result, nonlinear_result) -> None:
    """Print a human-readable comparison of the linear and nonlinear apex deflections."""
    print("Finite Element Toolkit")
    print("Version 16 -- Large-Displacement (Shallow) Truss")
    print("=" * 40)

    print(f"\nRise: {RISE} m, span: {SPAN} m, apex load: {APEX_LOAD:.3e} N")

    linear_uy = linear_result.displacement(2, Y)
    nonlinear_uy = nonlinear_result.displacement(2, Y)
    print(f"\nLinear apex deflection:              {linear_uy: .6e} m")
    print(f"Geometrically nonlinear deflection:  {nonlinear_uy: .6e} m")
    print(f"Ratio (nonlinear / linear):          {nonlinear_uy / linear_uy: .3f}")

    iteration_counts = [int(n) for n in nonlinear_result.iteration_counts()]
    print(f"\nNonlinear analysis converged: {nonlinear_result.converged}")
    print(f"Newton-Raphson iterations per load step: {iteration_counts}")
    print(
        "\nThe iteration count climbing toward the final steps is the honest "
        "signature of this shallow truss approaching its snap-through limit "
        "point -- see tests/validation/test_geometric_truss_large_displacement.py."
    )


if __name__ == "__main__":
    main()
