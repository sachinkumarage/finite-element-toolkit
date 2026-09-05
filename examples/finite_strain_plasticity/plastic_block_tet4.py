"""Example: a TET4 element driven into finite-strain J2 plasticity, Version 18.

Extends examples/tet4_j2_plasticity.py (Version 15, small-strain) to the
Version 18 finite-strain material, mirroring examples/nonlinear_tet4.py's
(Version 16) Total Lagrangian, geometrically nonlinear Newton-Raphson
setup: the element's own ``material`` is a placeholder LinearElastic3D,
and J2FiniteStrainPlasticity3D is supplied separately to
``NonlinearAnalysis(geometric_nonlinearity=True)``, driven through
incremental load stepping past yield.
"""

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.geometric_nonlinear import tet4_deformation_gradient
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.materials import J2FiniteStrainPlasticity3D, LinearElastic3D
from femtoolkit.mesh import Mesh, Node, Tet4Element3D
from femtoolkit.results.nonlinear_result import NonlinearAnalysisResult

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
YIELD_STRESS = 250e6  # Pa
HARDENING_MODULUS = 10e9  # Pa
TIP_LOAD = 3.0e8  # N, large enough to drive the element well past yield
LOAD_STEPS = 20


def main() -> None:
    """Build, solve, and report a single TET4 element's finite-strain plastic response."""
    placeholder_material = LinearElastic3D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, density=7850.0
    )
    plastic_material = J2FiniteStrainPlasticity3D(
        youngs_modulus=YOUNGS_MODULUS,
        poisson_ratio=POISSON_RATIO,
        yield_stress=YIELD_STRESS,
        hardening_modulus=HARDENING_MODULUS,
    )

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=0.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=0.0, z=1.0)
    tet = Tet4Element3D(id=1, nodes=(node_1, node_2, node_3, node_4), material=placeholder_material)

    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4):
        mesh.add_node(node)
    mesh.add_element(tet)

    settings = NonlinearSolverSettings(load_steps=LOAD_STEPS, tolerance=1e-9, max_iterations=50)
    analysis = NonlinearAnalysis(
        mesh, {tet.id: plastic_material}, settings, geometric_nonlinearity=True
    )
    for node_id in (1, 3, 4):
        for dof in (TranslationDOF.X, TranslationDOF.Y, TranslationDOF.Z):
            analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))
    analysis.add_load(NodalLoad(2, TranslationDOF.X, TIP_LOAD))

    result = analysis.solve()
    print_summary(mesh, tet, plastic_material, result)


def print_summary(
    mesh: Mesh,
    tet: Tet4Element3D,
    plastic_material: J2FiniteStrainPlasticity3D,
    result: NonlinearAnalysisResult,
) -> None:
    """Print a human-readable summary of the load-stepped Newton-Raphson solution."""
    print("Finite Element Toolkit")
    print("Version 18 -- TET4 Finite-Strain J2 Plasticity via Newton-Raphson")
    print("=" * 66)

    print(f"\nMesh: {len(mesh.nodes)} nodes, {len(mesh.elements)} TET4 element(s)")
    print(f"Converged: {result.converged}")
    print(f"Iterations per load step: {[int(n) for n in result.iteration_counts()]}")

    print("\nFinal nodal displacements:")
    for node in mesh.nodes:
        ux, uy, uz = result.node_displacement(node.id)
        print(f"    Node {node.id}: ux={ux:.6e}, uy={uy:.6e}, uz={uz:.6e} m")

    print("\nLoad-step-by-load-step material state (element 1):")
    for step in result.step_results:
        state = step.element_states[1].states[0]
        displacements = []
        for node in mesh.nodes:
            for dof in (TranslationDOF.X, TranslationDOF.Y, TranslationDOF.Z):
                displacements.append(step.displacement[result.dof_map.global_index(node.id, dof)])
        deformation_gradient = tet4_deformation_gradient(tet, displacements)
        cauchy = plastic_material.cauchy_stress(state, deformation_gradient)
        print(
            f"    load_factor={step.load_factor:.3f}  sigma_xx(Cauchy)={cauchy[0, 0]:.4e} Pa  "
            f"yielded={state.yielded}  alpha={state.hardening_variable:.6e}"
        )

    final_state = result.step_results[-1].element_states[1].states[0]
    print(f"\nFinal equivalent plastic strain (alpha): {final_state.hardening_variable:.6e}")
    print(
        "Final plastic deformation gradient (Fp) -- deviation from identity is the "
        "permanent, unrecoverable shape change:"
    )
    print(final_state.plastic_deformation_gradient)


if __name__ == "__main__":
    main()
