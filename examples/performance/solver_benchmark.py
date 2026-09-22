"""Example: solver + execution benchmark, Version 27.

Uses :func:`femtoolkit.performance.run_benchmark` to compare four
end-to-end configurations of a real ``StaticLinearAnalysis`` solve --
dense/serial (the Version 1-25 default), sparse direct/serial (Version
26), sparse direct/parallel element computation (Version 27), and
Conjugate Gradient/parallel -- on the same cantilever mesh, reporting
full :class:`~femtoolkit.performance.profiler.PerformanceReport`
diagnostics (element, assembly, and solve time; DOFs; non-zero count;
iterations where applicable) for each.

This demonstrates the benchmarking framework itself (``BenchmarkResult``,
:func:`~femtoolkit.performance.format_report`) rather than a solver
recommendation -- which configuration is fastest depends on mesh size,
boundary-condition count, and matrix conditioning, and is not fixed
across problems (see ``docs/solvers.md`` and ``docs/performance.md``).
"""

from __future__ import annotations

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.execution import ExecutionConfig
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh.generator import create_quad_mesh
from femtoolkit.performance import BenchmarkResult, format_report, run_benchmark
from femtoolkit.solvers import ConjugateGradientSolver, SparseDirectSolver

X = TranslationDOF.X
Y = TranslationDOF.Y


def _build_analysis(solver, execution) -> StaticLinearAnalysis:
    material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    mesh = create_quad_mesh(width=6.0, height=1.5, nx=60, ny=15, material=material, thickness=0.02)
    analysis = StaticLinearAnalysis(mesh, solver=solver, execution=execution)
    for node in mesh.nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(BoundaryCondition(node.id, X, 0.0))
            analysis.add_boundary_condition(BoundaryCondition(node.id, Y, 0.0))
    for node in mesh.nodes:
        if node.x == 6.0 and node.y == 0.0:
            analysis.add_load(NodalLoad(node.id, Y, -20000.0))
    return analysis


def main() -> None:
    """Benchmark four dense/sparse/serial/parallel configurations of a real cantilever solve."""
    print("Finite Element Toolkit")
    print("Version 27 -- Solver + Execution Benchmark")
    print("=" * 45)

    cases = [
        ("Dense, Serial", None, None),
        ("Sparse Direct, Serial", SparseDirectSolver(), None),
        (
            "Sparse Direct, Parallel (4 workers)",
            SparseDirectSolver(),
            ExecutionConfig(mode="parallel", workers=4, backend="process"),
        ),
        (
            "Conjugate Gradient, Parallel (4 workers)",
            ConjugateGradientSolver(tolerance=1e-10, max_iterations=2000),
            ExecutionConfig(mode="parallel", workers=4, backend="process"),
        ),
    ]

    results: list[BenchmarkResult] = []
    for name, solver, execution in cases:
        analysis = _build_analysis(solver, execution)
        analysis.solve()
        results.append(run_benchmark(name, lambda a=analysis: a.last_performance_report))

    for result in results:
        print()
        print(format_report(result.report))

    print(
        "\nNote: this is one mesh size on one machine -- relative ordering of these four "
        "configurations is not guaranteed to hold at a different scale. Re-run "
        "scaling_study.py to see how element-computation speedup changes with mesh size."
    )


if __name__ == "__main__":
    main()
