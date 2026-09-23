"""Example: comparing direct/iterative solver solutions and residual history (Version 29).

**Engineering problem/procedure.** The same cantilever plate is solved
three ways -- dense direct, sparse direct, and Conjugate Gradient (with
residual-history tracking enabled) -- and the resulting displacement
fields are compared for numerical consistency
(:func:`~femtoolkit.verification.solver_verification.compare_solver_solutions`).
The CG solve's residual history is also plotted
(:func:`~femtoolkit.verification.plots.plot_residual_history`).
"""

from __future__ import annotations

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh.generator import create_quad_mesh
from femtoolkit.solvers import ConjugateGradientSolver, DenseDirectSolver, SparseDirectSolver
from femtoolkit.verification.plots import plot_residual_history
from femtoolkit.verification.solver_verification import (
    compare_solver_solutions,
    solver_convergence_record,
)

X = TranslationDOF.X
Y = TranslationDOF.Y


def _build(solver):
    material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    mesh = create_quad_mesh(width=4.0, height=1.0, nx=20, ny=6, material=material, thickness=0.02)
    analysis = StaticLinearAnalysis(mesh, solver=solver)
    for node in mesh.nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(BoundaryCondition(node.id, X, 0.0))
            analysis.add_boundary_condition(BoundaryCondition(node.id, Y, 0.0))
    for node in mesh.nodes:
        if node.x == 4.0 and node.y == 0.0:
            analysis.add_load(NodalLoad(node.id, Y, -10000.0))
    return analysis


def main() -> None:
    """Compare dense, sparse, and CG solver solutions and plot the CG residual history."""
    print("Finite Element Toolkit")
    print("Version 29 -- Solver Comparison & Convergence Verification")
    print("=" * 60)

    dense_result = _build(DenseDirectSolver()).solve()
    sparse_result = _build(SparseDirectSolver()).solve()

    cg_analysis = _build(
        ConjugateGradientSolver(tolerance=1e-10, max_iterations=2000, track_residual_history=True)
    )
    cg_result = cg_analysis.solve()

    for name, result in (("Sparse Direct", sparse_result), ("Conjugate Gradient", cg_result)):
        comparison = compare_solver_solutions(
            f"Dense vs. {name}", dense_result.displacements, result.displacements
        )
        print(f"\n{comparison.message}")

    record = solver_convergence_record(cg_analysis.last_solver_result)
    print(f"\nConjugate Gradient: {record.iterations} iterations, converged={record.converged}")

    residual_history = cg_analysis.last_solver_result.diagnostics["residual_history"]
    figure = plot_residual_history(residual_history, solver_name="Conjugate Gradient")
    figure.savefig("examples/verification/solver_residual_history.png", dpi=100)
    print("Saved solver_residual_history.png")


if __name__ == "__main__":
    main()
