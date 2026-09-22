"""Example: serial vs. parallel element computation, Version 27.

Solves the same real FEA problem (a cantilever Q4 plate under a tip load)
with element-level stiffness computation run three ways -- serial (the
default, matching every prior version), parallel across worker
*processes* (:class:`~femtoolkit.execution.parallel.ParallelExecutor`,
``backend="process"``), and parallel across worker *threads*
(``backend="thread"``) -- and compares element-computation time and the
resulting displacement solution.

This is a genuine, honest comparison, not a demonstration engineered to
show a speedup: for this toolkit's Q4 element (a closed-form 2x2-Gauss
stiffness matrix, a few microseconds of real work per element), the
fixed per-task overhead of dispatching work to a worker process or
thread is often larger than the work itself, so parallel execution can
easily be *slower* than serial at the mesh sizes most engineering models
actually use. The example prints whatever numbers this run actually
produced. See ``scaling_study.py`` for where a real (if modest) speedup
does appear, and ``docs/performance.md`` for the full explanation.
"""

from __future__ import annotations

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.assembly import assemble_global_stiffness
from femtoolkit.analysis.dof import DOFMap
from femtoolkit.analysis.parallel_assembly import compute_stiffness_contributions
from femtoolkit.analysis.system import LinearSystem, build_force_vector, solve
from femtoolkit.execution import ExecutionConfig, create_executor
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh.generator import create_quad_mesh
from femtoolkit.performance import ELEMENT, Profiler

X = TranslationDOF.X
Y = TranslationDOF.Y


def main() -> None:
    """Assemble and solve a real cantilever mesh via serial, process, and thread execution."""
    print("Finite Element Toolkit")
    print("Version 27 -- Serial vs. Parallel Element Computation")
    print("=" * 55)

    material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    mesh = create_quad_mesh(width=6.0, height=1.5, nx=60, ny=15, material=material, thickness=0.02)
    print(f"\nMesh: {len(mesh.nodes)} nodes, {len(mesh.elements)} Q4 elements")

    dof_map = DOFMap(node_ids=[n.id for n in mesh.nodes], dofs_per_node=2)
    left_nodes = [n for n in mesh.nodes if n.x == 0.0]
    right_nodes = [n for n in mesh.nodes if n.x == 6.0 and n.y == 0.0]
    bcs = [BoundaryCondition(node.id, X, 0.0) for node in left_nodes] + [
        BoundaryCondition(node.id, Y, 0.0) for node in left_nodes
    ]
    loads = [NodalLoad(node.id, Y, -20000.0) for node in right_nodes]
    forces = build_force_vector(dof_map, loads)

    configs = [
        ("Serial", ExecutionConfig(mode="serial")),
        (
            "Parallel (process, 4 workers)",
            ExecutionConfig(mode="parallel", workers=4, backend="process"),
        ),
        (
            "Parallel (thread, 4 workers)",
            ExecutionConfig(mode="parallel", workers=4, backend="thread"),
        ),
    ]

    solutions = {}
    print("\n--- Element Computation ---")
    for name, config in configs:
        executor = create_executor(config)
        profiler = Profiler()
        with profiler, profiler.stage(ELEMENT):
            contributions = compute_stiffness_contributions(mesh.elements, executor)
        report = profiler.report(execution_mode=config.mode)
        print(f"{name:32s}: element_time={report.element_time:.4f} s")

        stiffness = assemble_global_stiffness(dof_map, contributions)
        system = LinearSystem(
            dof_map=dof_map, stiffness=stiffness, forces=forces, boundary_conditions=bcs
        )
        solutions[name] = solve(system)

    print("\n--- Numerical Agreement (vs. Serial) ---")
    baseline = solutions["Serial"]
    for name in ("Parallel (process, 4 workers)", "Parallel (thread, 4 workers)"):
        difference = abs(baseline - solutions[name]).max()
        print(f"{name:32s}: max |difference| = {difference:.3e}")

    print(
        "\nNote: at this mesh size, per-task dispatch overhead (pickling for the process "
        "backend, GIL contention for the thread backend) commonly outweighs the actual "
        "per-element computation for this toolkit's Q4 element -- parallel is not "
        "guaranteed to be faster here, and the numbers above are this run's actual, "
        "unmodified results. A genuine (if modest) speedup does appear at much larger "
        "element counts -- see scaling_study.py."
    )


if __name__ == "__main__":
    main()
