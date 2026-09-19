"""Example: dense vs. sparse solver comparison, Version 26.

Solves the same real FEA problem (a cantilever Q4 plate under a tip
load) three ways -- the existing dense direct solver
(:class:`~femtoolkit.solvers.dense.DenseDirectSolver`, numerically
identical to the toolkit's original
:func:`~femtoolkit.analysis.system.solve`), the new sparse direct
solver (:class:`~femtoolkit.solvers.sparse.SparseDirectSolver`), and
the new Conjugate Gradient iterative solver
(:class:`~femtoolkit.solvers.iterative.ConjugateGradientSolver`) --
comparing matrix size, non-zero count, matrix density, solve time, and
the numerical difference between solutions.

This is a genuine, honest comparison: sparse solve time is not
guaranteed to beat dense solve time at every problem size (factorization
overhead and Python-level bookkeeping can dominate for a small enough
system), so the example prints whatever numbers this run actually
produced rather than asserting a fixed ordering. Sparse storage's one
*unconditional* payoff is memory -- an ``O(nnz)`` rather than ``O(N^2)``
footprint, with ``nnz`` growing much more slowly than ``N^2`` as a
structured mesh is refined.
"""

import time

import numpy as np

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.assembly import ElementStiffnessContribution, assemble_global_stiffness
from femtoolkit.analysis.dof import DOFMap
from femtoolkit.analysis.sparse_assembly import assemble_global_stiffness_sparse
from femtoolkit.analysis.system import LinearSystem, build_force_vector
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh.generator import create_quad_mesh
from femtoolkit.solvers import ConjugateGradientSolver, DenseDirectSolver, SparseDirectSolver

X = TranslationDOF.X
Y = TranslationDOF.Y


def main() -> None:
    """Assemble a real cantilever-plate stiffness matrix and compare solver strategies."""
    print("Finite Element Toolkit")
    print("Version 26 -- Dense vs. Sparse Solver Comparison")
    print("=" * 50)

    material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    mesh = create_quad_mesh(
        width=4.0, height=1.0, nx=40, ny=10, material=material, thickness=0.02
    )
    print(f"\nMesh: {len(mesh.nodes)} nodes, {len(mesh.elements)} Q4 elements")

    dof_map = DOFMap(node_ids=[n.id for n in mesh.nodes], dofs_per_node=2)
    contributions = [
        ElementStiffnessContribution(e.dof_keys(), e.stiffness_matrix) for e in mesh.elements
    ]

    t0 = time.perf_counter()
    dense_stiffness = assemble_global_stiffness(dof_map, contributions)
    dense_assembly_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    sparse_stiffness = assemble_global_stiffness_sparse(dof_map, contributions)
    sparse_assembly_time = time.perf_counter() - t0

    n = dof_map.total_dofs
    nnz = sparse_stiffness.nnz
    density = nnz / (n * n)
    dense_bytes = dense_stiffness.nbytes
    sparse_bytes = (
        sparse_stiffness.data.nbytes
        + sparse_stiffness.indices.nbytes
        + sparse_stiffness.indptr.nbytes
    )

    print(f"\nDegrees of freedom: {n}")
    print(f"Dense assembly time:  {dense_assembly_time:.4f} s")
    print(f"Sparse assembly time: {sparse_assembly_time:.4f} s")
    print(f"\nDense matrix memory:  {dense_bytes / 1024:.1f} KB ({n}x{n}, {n * n} entries)")
    print(f"Sparse matrix memory: {sparse_bytes / 1024:.1f} KB ({nnz} non-zero entries)")
    print(f"Sparse density: {density:.5f} ({density * 100:.3f}% non-zero)")
    print(f"Memory ratio (sparse/dense): {sparse_bytes / dense_bytes:.4f}")

    left_nodes = [n for n in mesh.nodes if n.x == 0.0]
    right_nodes = [n for n in mesh.nodes if n.x == 4.0 and n.y == 0.0]
    bcs = [BoundaryCondition(node.id, X, 0.0) for node in left_nodes] + [
        BoundaryCondition(node.id, Y, 0.0) for node in left_nodes
    ]
    loads = [NodalLoad(node.id, Y, -10000.0) for node in right_nodes]
    forces = build_force_vector(dof_map, loads)

    dense_system = LinearSystem(
        dof_map=dof_map, stiffness=dense_stiffness, forces=forces, boundary_conditions=bcs
    )
    sparse_system = LinearSystem(
        dof_map=dof_map, stiffness=sparse_stiffness, forces=forces, boundary_conditions=bcs
    )

    print("\n--- Solver Comparison ---")
    results = {}
    for name, solver, system in [
        ("Dense Direct", DenseDirectSolver(), dense_system),
        ("Sparse Direct", SparseDirectSolver(), sparse_system),
        (
            "Conjugate Gradient",
            ConjugateGradientSolver(tolerance=1e-10, max_iterations=2000),
            sparse_system,
        ),
    ]:
        result = solver.solve(system)
        results[name] = result
        iteration_info = (
            f", iterations={result.iterations}" if result.iterations is not None else ""
        )
        print(
            f"{name:20s}: solve_time={result.solve_time:.4f} s, "
            f"converged={result.converged}, "
            f"relative_residual={result.relative_residual:.3e}{iteration_info}"
        )

    print("\n--- Numerical Agreement (vs. Dense Direct) ---")
    baseline = results["Dense Direct"].solution
    for name in ("Sparse Direct", "Conjugate Gradient"):
        difference = np.abs(baseline - results[name].solution).max()
        print(f"{name:20s}: max |difference| = {difference:.3e}")

    print(
        "\nNote: relative solver speed depends on mesh size, boundary-condition count, "
        "and matrix conditioning -- results above are this run's actual, unmodified "
        "numbers, not a guaranteed ordering. What sparse storage guarantees "
        "unconditionally is memory: the ratio printed above scales down further, not up, "
        "as the mesh grows, since a structured mesh's non-zero fraction shrinks with N."
    )


if __name__ == "__main__":
    main()
