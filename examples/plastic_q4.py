"""Example: a nonlinear Q4 element with independently plastic Gauss points, Version 14.

Extends ``examples/nonlinear_quad.py`` from Version 13 (which validated
Q4's per-Gauss-point architecture with a purely linear
``ElasticMaterialAdapter``) to genuine hardening plasticity via
:class:`~femtoolkit.materials.hardening.DecoupledIsotropicHardeningAdapter2D`
(see ``examples/plastic_cst.py`` for the honesty note about this
adapter's scope: it is not a real multiaxial yield-surface model, only a
decoupled per-component one used to validate the element architecture).

A bending-like load (opposing tip forces) is used deliberately, so some
Gauss points yield while others stay elastic within the *same* element
at the *same* load step -- direct proof that material state, internal
force, and tangent stiffness are evaluated independently at each of the
four Gauss points, not shared or averaged.
"""

import numpy as np

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.materials import DecoupledIsotropicHardeningAdapter2D, LinearElastic2D
from femtoolkit.mesh import Mesh, Node, QuadElement2D
from femtoolkit.results.nonlinear_result import NonlinearAnalysisResult

YOUNGS_MODULUS = 210e9  # Pa
YIELD_STRESS = 50e6  # Pa
HARDENING_MODULUS = 20e9  # Pa
THICKNESS = 0.01  # m
TIP_LOAD = 3.0e5  # N, opposing forces at Node 2 (down) and Node 3 (up)
LOAD_STEPS = 6


def main() -> None:
    """Build, solve, and report per-Gauss-point plastic state for a single nonlinear Q4 element."""
    material = DecoupledIsotropicHardeningAdapter2D(
        youngs_modulus=YOUNGS_MODULUS,
        yield_stress=YIELD_STRESS,
        hardening_modulus=HARDENING_MODULUS,
    )
    placeholder_material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=0.3, formulation="plane_stress"
    )

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=1.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=1.0, z=0.0)
    quad = QuadElement2D(
        id=1,
        nodes=(node_1, node_2, node_3, node_4),
        material=placeholder_material,
        thickness=THICKNESS,
    )

    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4):
        mesh.add_node(node)
    mesh.add_element(quad)

    settings = NonlinearSolverSettings(load_steps=LOAD_STEPS, tolerance=1e-9, max_iterations=40)
    analysis = NonlinearAnalysis(mesh, {quad.id: material}, settings)
    analysis.add_boundary_condition(BoundaryCondition(1, TranslationDOF.X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, TranslationDOF.Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, TranslationDOF.X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, TranslationDOF.Y, 0.0))
    analysis.add_load(NodalLoad(2, TranslationDOF.Y, TIP_LOAD))
    analysis.add_load(NodalLoad(3, TranslationDOF.Y, -TIP_LOAD))
    result = analysis.solve()

    print_summary(mesh, result)


def print_summary(mesh: Mesh, result: NonlinearAnalysisResult) -> None:
    """Print a human-readable summary, including each Gauss point's plastic state."""
    print("Finite Element Toolkit")
    print("Version 14 -- Nonlinear Q4 Element With Independently Plastic Gauss Points")
    print("=" * 40)

    print(f"\nMesh:\n    {len(mesh.nodes)} nodes, {len(mesh.elements)} Q4 element(s)")
    print(f"\nConverged: {result.converged}")
    print(f"Iterations per load step: {[int(n) for n in result.iteration_counts()]}")

    print("\nFinal nodal displacements:")
    for node in mesh.nodes:
        ux, uy = result.node_displacement(node.id)
        print(f"    Node {node.id}: ux = {ux:.6e} m, uy = {uy:.6e} m")

    states = result.step_results[-1].element_states[1].states
    print("\nFinal Gauss-point plastic state (element 1, 4 independent points):")
    for index, state in enumerate(states):
        print(f"    Point {index}: stress={state.stress}")
        print(f"              plastic_strain={state.plastic_strain}")
        print(f"              alpha={state.hardening_variable}, yielded={state.yielded}")

    point_yielded = [bool(np.any(state.yielded)) for state in states]
    mixed_state = any(point_yielded) and not all(point_yielded)
    print(f"\nGauss points yielded: {point_yielded}")
    print(f"At least one yielded, and at least one stayed elastic: {mixed_state}")


if __name__ == "__main__":
    main()
