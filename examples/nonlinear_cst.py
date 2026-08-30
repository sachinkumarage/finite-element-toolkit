"""Example: a nonlinear CST element solved through NonlinearAnalysis, Version 13.

Demonstrates the full nonlinear continuum workflow for a CST element:

.. code-block:: text

    Material (ElasticMaterialAdapter) -> Nodes -> CSTElement2D -> Mesh
        -> NonlinearAnalysis (boundary conditions + loads + settings)
        -> Newton-Raphson load stepping -> LoadStepResult per increment

:class:`~femtoolkit.materials.nonlinear.ElasticMaterialAdapter` wraps the
element's ordinary linear material as a
:class:`~femtoolkit.materials.nonlinear.NonlinearMaterial` -- this is the
Version 13 validation strategy for continuum elements (see
:mod:`femtoolkit.analysis.nonlinear_elements`): proving the internal
force/tangent stiffness/residual machinery is correct by reproducing the
existing linear solver's answer exactly, since a real 2D plasticity
model (a multiaxial yield surface) is out of this version's scope.

Model:

    Node4 -------- Node3
     |          .   |
     |       .      |
     |    .         |
    Node1 -------- Node2

    Node 1 = (0, 0), Node 2 = (1, 0), Node 3 = (1, 1), Node 4 = (0, 1)
    Two triangles: (1, 2, 3) and (1, 3, 4)
    E = 210 GPa, v = 0.3 (plane stress), t = 0.01 m
    Node 1, Node 4: fixed (ux = uy = 0)
    Fx = 1e6 N each at Node 2 and Node 3, applied over 5 load steps
"""

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.materials import ElasticMaterialAdapter, LinearElastic2D
from femtoolkit.mesh import CSTElement2D, Mesh, Node
from femtoolkit.results.nonlinear_result import NonlinearAnalysisResult

YOUNGS_MODULUS = 210e9  # Pa
POISSON_RATIO = 0.3
THICKNESS = 0.01  # m
APPLIED_LOAD = 1.0e6  # N, at each of Node 2 and Node 3
LOAD_STEPS = 5


def main() -> None:
    """Build, solve, and report results for a two-triangle nonlinear CST patch."""
    linear_material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, formulation="plane_stress"
    )

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=1.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=1.0, z=0.0)
    lower = CSTElement2D(
        id=1, nodes=(node_1, node_2, node_3), material=linear_material, thickness=THICKNESS
    )
    upper = CSTElement2D(
        id=2, nodes=(node_1, node_3, node_4), material=linear_material, thickness=THICKNESS
    )

    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4):
        mesh.add_node(node)
    mesh.add_element(lower)
    mesh.add_element(upper)

    materials = {
        element.id: ElasticMaterialAdapter.from_linear_elastic_2d(linear_material)
        for element in (lower, upper)
    }
    settings = NonlinearSolverSettings(load_steps=LOAD_STEPS, tolerance=1e-10)
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
    """Print a human-readable summary of the nonlinear CST analysis."""
    print("Finite Element Toolkit")
    print("Version 13 -- Nonlinear CST Element")
    print("=" * 40)

    print(f"\nMesh:\n    {len(mesh.nodes)} nodes, {len(mesh.elements)} CST elements")
    print(f"\nConverged: {result.converged}")

    print(f"\n{'step':>4} {'load factor':>12} {'iterations':>11} {'residual norm':>14}")
    for i, step_result in enumerate(result.step_results):
        print(
            f"{i + 1:4d} {step_result.load_factor:12.3f} "
            f"{step_result.iterations:11d} {step_result.residual_norm:14.3e}"
        )

    print("\nFinal nodal displacements:")
    for node in mesh.nodes:
        ux, uy = result.node_displacement(node.id)
        print(f"    Node {node.id}: ux = {ux:.6e} m, uy = {uy:.6e} m")

    print("\nFinal element strain / stress (element 1, single-state CST):")
    print(f"    strain = {result.element_strain(1)}")
    print(f"    stress = {result.element_stress(1)} Pa")

    reactions = result.step_results[-1].reactions
    total_applied = 2.0 * APPLIED_LOAD
    equilibrium = reactions.sum() + total_applied
    print(f"\nEquilibrium check: sum(reactions) + sum(applied) = {equilibrium:.6e}")


if __name__ == "__main__":
    main()
