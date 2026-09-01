"""Example: a nonlinear CST element with hardening plasticity, Version 14.

Extends ``examples/nonlinear_cst.py`` from Version 13 (which validated
the CST nonlinear architecture with a purely linear
``ElasticMaterialAdapter``) to a genuinely path-dependent, hardening
material: :class:`~femtoolkit.materials.hardening.DecoupledIsotropicHardeningAdapter2D`.

That adapter applies :class:`~femtoolkit.materials.hardening.BilinearIsotropicHardeningMaterial1D`'s
exact scalar return map **independently to each of the 3 Voigt strain
components** -- a deliberate simplification for validating CST's
per-Gauss-point-equivalent architecture (internal force, tangent
stiffness, Newton-Raphson convergence against a genuinely changing
tangent), not a claim of physically accurate multiaxial plasticity (see
that class's docstring). Real multiaxial (J2/von Mises) plasticity is
out of scope through Version 14.

Model: the same two-triangle patch as ``examples/nonlinear_cst.py``,
loaded hard enough to yield.
"""

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.materials import DecoupledIsotropicHardeningAdapter2D, LinearElastic2D
from femtoolkit.mesh import CSTElement2D, Mesh, Node
from femtoolkit.results.nonlinear_result import NonlinearAnalysisResult

YOUNGS_MODULUS = 210e9  # Pa
YIELD_STRESS = 150e6  # Pa
HARDENING_MODULUS = 20e9  # Pa
THICKNESS = 0.01  # m
APPLIED_LOAD = 3.0e6  # N, at each of Node 2 and Node 3
LOAD_STEPS = 10


def main() -> None:
    """Build, solve, and report results for a two-triangle nonlinear CST patch, with plasticity."""
    material = DecoupledIsotropicHardeningAdapter2D(
        youngs_modulus=YOUNGS_MODULUS,
        yield_stress=YIELD_STRESS,
        hardening_modulus=HARDENING_MODULUS,
    )

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=1.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=1.0, z=0.0)
    # The CST elements still need an (unused, for nonlinear analysis) linear
    # material to construct -- nonlinear analysis reads only the separate
    # `materials` mapping passed to NonlinearAnalysis, never `element.material`.
    placeholder_material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=0.3, formulation="plane_stress"
    )
    lower = CSTElement2D(
        id=1, nodes=(node_1, node_2, node_3), material=placeholder_material, thickness=THICKNESS
    )
    upper = CSTElement2D(
        id=2, nodes=(node_1, node_3, node_4), material=placeholder_material, thickness=THICKNESS
    )

    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4):
        mesh.add_node(node)
    mesh.add_element(lower)
    mesh.add_element(upper)

    materials = {element.id: material for element in (lower, upper)}
    settings = NonlinearSolverSettings(load_steps=LOAD_STEPS, tolerance=1e-9, max_iterations=40)
    analysis = NonlinearAnalysis(mesh, materials, settings)
    analysis.add_boundary_condition(BoundaryCondition(1, TranslationDOF.X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, TranslationDOF.Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, TranslationDOF.X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, TranslationDOF.Y, 0.0))
    analysis.add_load(NodalLoad(2, TranslationDOF.X, APPLIED_LOAD))
    analysis.add_load(NodalLoad(3, TranslationDOF.X, APPLIED_LOAD))
    result = analysis.solve()

    print_summary(mesh, result)


def print_summary(mesh: Mesh, result: NonlinearAnalysisResult) -> None:
    """Print a human-readable summary of the nonlinear plastic CST analysis."""
    print("Finite Element Toolkit")
    print("Version 14 -- Nonlinear CST Element With Hardening Plasticity")
    print("=" * 40)

    print(f"\nMesh:\n    {len(mesh.nodes)} nodes, {len(mesh.elements)} CST elements")
    print(f"\nConverged: {result.converged}")
    print(f"Iterations per load step: {[int(n) for n in result.iteration_counts()]}")

    print("\nFinal nodal displacements:")
    for node in mesh.nodes:
        ux, uy = result.node_displacement(node.id)
        print(f"    Node {node.id}: ux = {ux:.6e} m, uy = {uy:.6e} m")

    print("\nFinal element 1 state (Voigt components: x, y, xy):")
    print(f"    strain          = {result.element_strain(1)}")
    print(f"    stress          = {result.element_stress(1)} Pa")
    print(f"    plastic strain  = {result.element_plastic_strain(1)}")
    print(f"    hardening alpha = {result.element_hardening_variable(1)}")
    print(f"    yielded         = {result.element_yielded(1)}")


if __name__ == "__main__":
    main()
