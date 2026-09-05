"""Example: a HEX8 element driven into finite-strain J2 plasticity, Version 18.

Extends examples/hex8_j2_plasticity.py (Version 15, small-strain) and
examples/neo_hookean_block.py (Version 17, elastic hyperelasticity) to the
Version 18 finite-strain plastic material: a HEX8 block pulled well past
its yield point under Total Lagrangian geometric nonlinearity, reporting
each Gauss point's independent plastic state -- unlike the single-point
TET4 example (plastic_block_tet4.py), demonstrating that yielding can
progress independently at different points within one element.
"""

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.geometric_nonlinear import hex8_deformation_gradients
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.materials import J2FiniteStrainPlasticity3D, LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.results.nonlinear_result import NonlinearAnalysisResult

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
YIELD_STRESS = 250e6  # Pa
HARDENING_MODULUS = 10e9  # Pa
LOAD_PER_NODE = 8.0e7  # N, applied at each of the 4 loaded-face nodes
LOAD_STEPS = 20

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]
_FIXED_FACE = (1, 4, 5, 8)  # x = 0 face
_LOADED_FACE = (2, 3, 6, 7)  # x = 1 face


def main() -> None:
    """Build, solve, and report a HEX8 block's finite-strain plastic response."""
    placeholder_material = LinearElastic3D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, density=7850.0
    )
    plastic_material = J2FiniteStrainPlasticity3D(
        youngs_modulus=YOUNGS_MODULUS,
        poisson_ratio=POISSON_RATIO,
        yield_stress=YIELD_STRESS,
        hardening_modulus=HARDENING_MODULUS,
    )

    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder_material)

    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    settings = NonlinearSolverSettings(load_steps=LOAD_STEPS, tolerance=1e-9, max_iterations=40)
    analysis = NonlinearAnalysis(
        mesh, {hexa.id: plastic_material}, settings, geometric_nonlinearity=True
    )
    for node_id in _FIXED_FACE:
        for dof in (TranslationDOF.X, TranslationDOF.Y, TranslationDOF.Z):
            analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))
    for node_id in _LOADED_FACE:
        analysis.add_load(NodalLoad(node_id, TranslationDOF.X, LOAD_PER_NODE))

    result = analysis.solve()
    print_summary(mesh, hexa, plastic_material, result)


def print_summary(
    mesh: Mesh,
    hexa: Hex8Element3D,
    plastic_material: J2FiniteStrainPlasticity3D,
    result: NonlinearAnalysisResult,
) -> None:
    """Print a human-readable summary, including each Gauss point's final plastic state."""
    print("Finite Element Toolkit")
    print("Version 18 -- HEX8 Finite-Strain J2 Plasticity via Newton-Raphson")
    print("=" * 66)

    print(f"\nMesh: {len(mesh.nodes)} nodes, {len(mesh.elements)} HEX8 element(s)")
    print(f"Converged: {result.converged}")
    print(f"Iterations per load step: {[int(n) for n in result.iteration_counts()]}")

    print("\nFinal nodal displacements:")
    for node in mesh.nodes:
        ux, uy, uz = result.node_displacement(node.id)
        print(f"    Node {node.id}: ux={ux:.6e}, uy={uy:.6e}, uz={uz:.6e} m")

    final_displacements = []
    for node in mesh.nodes:
        for dof in (TranslationDOF.X, TranslationDOF.Y, TranslationDOF.Z):
            final_displacements.append(
                result.step_results[-1].displacement[result.dof_map.global_index(node.id, dof)]
            )
    deformation_gradients = hex8_deformation_gradients(hexa, final_displacements)

    print("\nFinal per-Gauss-point plastic state:")
    for gauss_point, deformation_gradient_tensor in enumerate(deformation_gradients):
        state = result.element_state(1, gauss_point=gauss_point)
        cauchy = plastic_material.cauchy_stress(state, deformation_gradient_tensor)
        print(
            f"    Point {gauss_point}: sigma_xx(Cauchy)={cauchy[0, 0]:.4e} Pa, "
            f"yielded={state.yielded}, alpha={state.hardening_variable:.6e}"
        )

    total_load = LOAD_PER_NODE * len(_LOADED_FACE)
    nominal_stress = total_load / 1.0  # unit reference cross-section area
    print(f"\nNominal reference-area stress (total load / reference area): {nominal_stress:.4e} Pa")


if __name__ == "__main__":
    main()
