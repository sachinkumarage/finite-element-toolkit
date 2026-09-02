"""Example: a HEX8 element with independently plastic Gauss points, Version 15.

Extends examples/hex8_linear_elastic.py to J2 plasticity, mirroring
examples/plastic_q4.py (Version 14): a bending-like load (opposing forces
on the top and bottom of the free face) is used deliberately, so some of
the 8 Gauss points yield while others stay elastic within the *same*
element at the *same* load step -- direct proof that material state,
internal force, and tangent stiffness are evaluated independently at each
Gauss point, not shared or averaged (spec section 27).
"""

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.continuum.stress import von_mises_3d
from femtoolkit.materials import J2Plasticity3D, LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.results.nonlinear_result import NonlinearAnalysisResult

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
YIELD_STRESS = 80e6  # Pa
HARDENING_MODULUS = 5e9  # Pa
TIP_LOAD = 9.0e7  # N, opposing forces at the free face's top/bottom edges
LOAD_STEPS = 15


def main() -> None:
    """Build, solve, and report per-Gauss-point plastic state for a nonlinear HEX8 element."""
    placeholder_material = LinearElastic3D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, density=7850.0
    )
    j2_material = J2Plasticity3D(
        youngs_modulus=YOUNGS_MODULUS,
        poisson_ratio=POISSON_RATIO,
        yield_stress=YIELD_STRESS,
        hardening_modulus=HARDENING_MODULUS,
    )

    # A long, thin cantilever-like beam so a tip moment produces meaningfully
    # different Gauss-point strains between the top and bottom fibers.
    coords = [
        (0.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (4.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (4.0, 0.0, 1.0),
        (4.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ]
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(coords))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder_material)

    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    settings = NonlinearSolverSettings(load_steps=LOAD_STEPS, tolerance=1e-9, max_iterations=50)
    analysis = NonlinearAnalysis(mesh, {hexa.id: j2_material}, settings)
    fixed_face = (1, 4, 5, 8)  # x = 0 face (cantilever root)
    for node_id in fixed_face:
        for dof in (TranslationDOF.X, TranslationDOF.Y, TranslationDOF.Z):
            analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))
    # Opposing Y forces at the free tip's top (node 3, 7) and bottom (node 2, 6)
    # edges create a bending moment about Z, distorting the element non-uniformly.
    analysis.add_load(NodalLoad(3, TranslationDOF.Y, TIP_LOAD))
    analysis.add_load(NodalLoad(7, TranslationDOF.Y, TIP_LOAD))
    analysis.add_load(NodalLoad(2, TranslationDOF.Y, -TIP_LOAD))
    analysis.add_load(NodalLoad(6, TranslationDOF.Y, -TIP_LOAD))

    result = analysis.solve()
    print_summary(mesh, result)


def print_summary(mesh: Mesh, result: NonlinearAnalysisResult) -> None:
    """Print a human-readable summary, including each Gauss point's plastic state."""
    print("Finite Element Toolkit")
    print("Version 15 -- HEX8 J2 Plasticity, Independently Plastic Gauss Points")
    print("=" * 40)

    print(f"\nMesh: {len(mesh.nodes)} nodes, {len(mesh.elements)} HEX8 element(s)")
    print(f"Converged: {result.converged}")
    print(f"Iterations per load step: {[int(n) for n in result.iteration_counts()]}")

    states = result.step_results[-1].element_states[1].states
    print("\nFinal Gauss-point plastic state (element 1, 8 independent points):")
    for index, state in enumerate(states):
        von_mises = von_mises_3d(*state.stress)
        print(
            f"    Point {index}: von_mises={von_mises:.4e} Pa  yielded={state.yielded}  "
            f"alpha={state.hardening_variable:.6e}"
        )

    point_yielded = [bool(state.yielded) for state in states]
    mixed_state = any(point_yielded) and not all(point_yielded)
    print(f"\nGauss points yielded: {point_yielded}")
    print(f"At least one yielded, and at least one stayed elastic: {mixed_state}")


if __name__ == "__main__":
    main()
