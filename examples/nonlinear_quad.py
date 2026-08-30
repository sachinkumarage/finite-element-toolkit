"""Example: a nonlinear Q4 element with independent Gauss-point state, Version 13.

Demonstrates the same nonlinear continuum workflow as
``examples/nonlinear_cst.py``, but for a Q4 element -- highlighting the
one piece unique to Q4: **each of its four Gauss points maintains its
own, independent material state** (see
:mod:`femtoolkit.analysis.nonlinear_elements`'s docstring on
:class:`~femtoolkit.analysis.nonlinear_elements.NonlinearElementState`).
A bending-like load (opposing tip forces) is used deliberately, so the
four Gauss points visibly disagree on strain and stress -- proof they
are evaluated independently rather than sharing one averaged value.

Model: a single Q4 element, fixed along its left edge, loaded by
opposing Y-direction forces at its two right nodes (a bending-like
load), solved via
:class:`~femtoolkit.analysis.nonlinear_analysis.NonlinearAnalysis` with
:class:`~femtoolkit.materials.nonlinear.ElasticMaterialAdapter` (see
``nonlinear_cst.py`` for why the elastic adapter is the correct
validation material for continuum elements in this version).
"""

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.materials import ElasticMaterialAdapter, LinearElastic2D
from femtoolkit.mesh import Mesh, Node, QuadElement2D
from femtoolkit.results.nonlinear_result import NonlinearAnalysisResult

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
THICKNESS = 0.01  # m
TIP_LOAD = 5.0e4  # N, opposing forces at Node 2 (down) and Node 3 (up)
LOAD_STEPS = 4


def main() -> None:
    """Build, solve, and report Gauss-point results for a single nonlinear Q4 element."""
    linear_material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, formulation="plane_stress"
    )

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=1.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=1.0, z=0.0)
    quad = QuadElement2D(
        id=1, nodes=(node_1, node_2, node_3, node_4), material=linear_material, thickness=THICKNESS
    )

    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4):
        mesh.add_node(node)
    mesh.add_element(quad)

    materials = {quad.id: ElasticMaterialAdapter.from_linear_elastic_2d(linear_material)}
    settings = NonlinearSolverSettings(load_steps=LOAD_STEPS)
    analysis = NonlinearAnalysis(mesh, materials, settings)
    analysis.add_boundary_condition(BoundaryCondition(1, TranslationDOF.X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, TranslationDOF.Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, TranslationDOF.X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, TranslationDOF.Y, 0.0))
    analysis.add_load(NodalLoad(2, TranslationDOF.Y, TIP_LOAD))
    analysis.add_load(NodalLoad(3, TranslationDOF.Y, -TIP_LOAD))
    result = analysis.solve()

    print_summary(mesh, result)


def print_summary(mesh: Mesh, result: NonlinearAnalysisResult) -> None:
    """Print a human-readable summary of the nonlinear Q4 analysis, including Gauss-point states."""
    print("Finite Element Toolkit")
    print("Version 13 -- Nonlinear Q4 Element (Independent Gauss-Point State)")
    print("=" * 40)

    print(f"\nMesh:\n    {len(mesh.nodes)} nodes, {len(mesh.elements)} Q4 element(s)")
    print(f"\nConverged: {result.converged}")
    print(f"Iterations per load step: {[int(n) for n in result.iteration_counts()]}")

    print("\nFinal nodal displacements:")
    for node in mesh.nodes:
        ux, uy = result.node_displacement(node.id)
        print(f"    Node {node.id}: ux = {ux:.6e} m, uy = {uy:.6e} m")

    states = result.step_results[-1].element_states[1].states
    print("\nFinal Gauss-point states (element 1, 4 independent points):")
    print(f"{'point':>6} {'strain (x,y,xy)':>42} {'stress (x,y,xy)':>42}")
    for i, state in enumerate(states):
        strain_str = ", ".join(f"{v: .4e}" for v in state.strain)
        stress_str = ", ".join(f"{v: .4e}" for v in state.stress)
        print(f"{i:6d} [{strain_str}] [{stress_str}]")

    distinct_objects = len({id(state) for state in states})
    print(f"\nDistinct Gauss-point state objects: {distinct_objects} (expected 4)")


if __name__ == "__main__":
    main()
