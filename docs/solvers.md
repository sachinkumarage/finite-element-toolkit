# Advanced Solver Infrastructure & Sparse Linear Algebra (Version 26)

Sparse matrices, a solver abstraction (dense direct, sparse direct, and
Conjugate Gradient), convergence monitoring, solver diagnostics, and
performance measurement -- laid over the toolkit's existing element
formulations and boundary-condition handling without changing either.
This document is the detailed technical guide; see the main
[README](../README.md#version-26) for a shorter overview and how
Version 26 fits into the toolkit's history.

## The global FEA system

A typical linear FEA problem is

$$
\mathbf{K}\mathbf{u} = \mathbf{F}
$$

where `K` is the global stiffness matrix, `u` the unknown displacement
vector, and `F` the global force vector (the thermal analogue,
`K_T T = Q`, is the same linear-algebra problem with a differently
named stiffness/conductivity matrix). For a mesh with `N` degrees of
freedom, `K` is usually **sparse**: each element only connects nodes
that are physically close to each other, so a given global row has
non-zero entries only in the columns belonging to that node's own
neighboring elements -- most of the `N^2` possible entries are exactly
zero.

```text
Dense (all N^2 entries stored):          Sparse (only nnz non-zero entries):
[████████████████]                       [██░░░░░░░░░░░░██]
[████████████████]                       [███░░░░░░░░░░░░░]
[████████████████]                       [░░███░░░░░░░░░░░]
[████████████████]                       [░░░░███░░░░░░░░░]
```

Storing only the non-zero entries turns an `O(N^2)` memory footprint
into roughly `O(nnz)`, where `nnz` grows much more slowly than `N^2` as
a structured mesh is refined -- this is what makes solving substantially
larger models practical, and is the entire motivation for this version.

## Dense vs. sparse trade-offs

- **Dense** -- simplest to reason about, no assembly-format choices,
  remains the default and stays fully supported (small systems,
  debugging, verification, every prior example). Memory and, past a
  certain size, solve time scale with `N^2`.
- **Sparse** -- memory scales with `nnz`, not `N^2` -- an unconditional
  win as a mesh grows. Solve time is *not* unconditionally faster at
  every size (factorization/bookkeeping overhead can dominate a small
  system); see `examples/solvers/sparse_solver_comparison.py` for a
  real, unmodified measurement rather than a claimed ordering.

## Sparse matrix formats

This toolkit uses SciPy's sparse module (`scipy.sparse`, part of the
`scipy` dependency the toolkit has required since Version 11 -- no new
dependency was needed for Version 26) rather than a custom sparse
matrix implementation, per spec section 5's explicit preference for an
established numerical backend.

- **COO** (coordinate format: parallel `row`/`col`/`value` arrays, one
  entry per triplet) -- used *during assembly*
  (`femtoolkit.analysis.sparse_assembly`). Appending a triplet is O(1),
  and SciPy automatically **sums duplicate `(row, col)` triplets**
  during the COO -> CSR conversion, giving the same "elements sharing a
  DOF get their contributions summed" behavior the dense scatter-add
  loop provides -- verified directly in `tests/test_sparse_assembly.py`.
- **CSR** (compressed sparse row) -- used for *solving*: fast row
  slicing (eliminating boundary-condition rows) and fast
  matrix-vector products (every solver, direct or iterative, needs
  these).
- **CSC** (compressed sparse column) -- used by
  `scipy.sparse.linalg.spsolve`'s underlying SuperLU factorization,
  which prefers column-major storage; `SparseDirectSolver` converts to
  CSC only at the point of solving.

## Sparse global assembly

`femtoolkit.analysis.sparse_assembly.assemble_global_stiffness_sparse`/
`assemble_global_mass_sparse` assemble **directly** into sparse form --
never building a dense array and converting it (spec section 6
explicitly warns against repeated dense/sparse conversion). Both are
mathematically equivalent to their existing dense counterparts
(`femtoolkit.analysis.assembly.assemble_global_stiffness`/
`assemble_global_mass`, unchanged since Version 2/11) for the same
inputs -- verified by comparing `K_dense` against `K_sparse.toarray()`
within floating-point tolerance for real element contributions (Q4,
CST) in `tests/test_sparse_assembly.py`. A shared private validator,
`femtoolkit.analysis.assembly._validate_contribution_shape`, is reused
by both the dense and sparse assemblers so a malformed contribution
reports the identical error either way.

## The solver abstraction

```text
LinearSolver (femtoolkit.solvers.base)
|-- DenseDirectSolver        (femtoolkit.solvers.dense)
|-- SparseDirectSolver       (femtoolkit.solvers.sparse)
`-- ConjugateGradientSolver  (femtoolkit.solvers.iterative)
```

Every concrete solver implements one method:

```python
result = solver.solve(system)  # system: femtoolkit.analysis.system.LinearSystem
```

reusing the existing `LinearSystem` container (Version 2) rather than
inventing a second one -- `LinearSystem.stiffness` may be a dense
`numpy.ndarray` (as it always was) or, since this version, a
`scipy.sparse.spmatrix`; `LinearSystem` itself needed no code changes,
since its own validation only inspects `.shape`, which both expose
identically.

### Dense direct (`DenseDirectSolver`)

Wraps `numpy.linalg.solve` on the free-free reduced system --
numerically the exact same computation
`femtoolkit.analysis.system.solve` (unchanged) has always performed.
Remains the default and fully available for small systems, debugging,
and verification (every sparse/iterative result in this version is
checked against it).

### Sparse direct (`SparseDirectSolver`)

Solves via `scipy.sparse.linalg.spsolve` (SuperLU) on the free-free
reduced system, without ever converting the matrix to dense form.

### Conjugate Gradient (`ConjugateGradientSolver`)

For a symmetric positive-definite (SPD) matrix `A` -- every stiffness
matrix this toolkit assembles from a properly constrained, physically
valid linear elastic or thermal conduction problem is SPD once boundary
conditions are eliminated -- solving `A x = b` is exactly equivalent to
minimizing the quadratic form

$$
f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T\mathbf{A}\mathbf{x} - \mathbf{b}^T\mathbf{x}
$$

(its unique minimum occurs exactly where its gradient, `Ax - b`,
vanishes). Conjugate Gradient minimizes this quadratic form by
searching a sequence of mutually `A`-conjugate directions, converging
in at most `N` iterations in exact arithmetic -- and, in floating-point
practice, to a useful tolerance in far fewer -- using only sparse
matrix-vector products, never an explicit factorization. This is why CG
is the standard choice for large, sparse SPD systems where a direct
factorization's fill-in would be too expensive, and why it is
*inappropriate* for a non-symmetric or indefinite matrix.

`ConjugateGradientSolver.check_symmetry` (default `True`) runs a cheap
`O(nnz)` `||A - A^T|| / ||A||` check before solving and raises
`InvalidSolverConfigurationError` if the matrix is not symmetric to
within tolerance -- spec section 15's "provide appropriate diagnostics
when the matrix does not satisfy the required assumptions." Only one
iterative method is implemented (spec section 11 explicitly scopes out
"an unnecessarily large collection"); CG alone covers this toolkit's
actual SPD systems.

## Convergence criteria

The residual of the reduced system that was actually solved:

$$
\mathbf{r} = \mathbf{b} - \mathbf{A}\mathbf{x}, \qquad
r_{rel} = \frac{\lVert \mathbf{r} \rVert}{\max(\lVert \mathbf{b} \rVert, \epsilon)}
$$

is computed identically for every solver (`femtoolkit.solvers.base.residual_norms`),
direct or iterative -- for a direct solver this is a verification
check; for CG it is the convergence criterion itself. `tolerance`
(relative residual) and `max_iterations` are both validated
(`InvalidSolverConfigurationError` for a non-positive or non-finite
value). `ConjugateGradientSolver.raise_on_non_convergence` (default
`True`) controls whether a non-converged solve raises
`SolverConvergenceError` or returns a `SolverResult` with
`converged=False` for inspection.

## Solver diagnostics: `SolverResult`

Every solver returns a `femtoolkit.solvers.results.SolverResult`
rather than a bare array:

```text
SolverResult
|-- solution            the solved vector
|-- converged           bool
|-- iterations          int, or None for a direct solve
|-- residual_norm        ||b - Ax|| on the reduced system
|-- relative_residual     residual_norm / max(||b||, eps)
|-- solve_time            wall-clock seconds inside the numerical routine
|-- solver_name           "Dense Direct" / "Sparse Direct" / "Conjugate Gradient"
`-- diagnostics            dofs, free_dofs, nnz, density (only what was actually computed)
```

## Singular-system detection

A dense or sparse LU factorization only raises an exception for an
*exactly* singular matrix (an exact zero pivot); a matrix that is
singular in exact arithmetic but has merely a very small pivot after
floating-point round-off -- exactly what a free rigid-body mechanism
with more than one under-constrained DOF produces -- factorizes and
"solves" without raising, returning a physically meaningless
huge-magnitude vector. `femtoolkit.solvers.base.check_direct_solve_residual`
catches this: after a direct solve, if the relative residual exceeds a
threshold many orders of magnitude looser than a genuinely solved
system's (`1e-4`, versus `1e-10`-`1e-15` for a real solve), it raises
`SingularSystemError` -- the toolkit's existing exception, reused
rather than duplicated. Verified directly against a real
under-constrained mesh in `tests/test_solvers.py`.

## Boundary conditions and sparse systems

The existing elimination approach (`femtoolkit.analysis.system.solve`,
Version 2, unchanged) partitions global DOFs into free and constrained
sets, moves the constrained DOFs' known contribution to the
right-hand side, and solves only the reduced free-free system.
`femtoolkit.solvers.base.partition_dofs`/`reduced_system` implement the
identical logic once, shared by every solver, dispatching on whether
`system.stiffness` is dense (`numpy.ix_` fancy indexing) or sparse
(CSR row-then-column indexing).

Multi-point constraints (the penalty method,
`femtoolkit.analysis.multi_point_constraint`, unchanged since its
introduction) gained a sparse analogue,
`apply_multi_point_constraints_sparse`: the identical penalty formula
and penalty-stiffness scaling, applied via `scipy.sparse.lil_matrix`
(efficient for incremental single-entry updates) and converted back to
CSR. Verified to match the dense function's output exactly in
`tests/test_multi_point_constraint.py`.

## Mechanical and thermal integration

`StaticLinearAnalysis.__init__` and `SteadyStateThermalAnalysis` both
gained an optional `solver: LinearSolver | None = None`
parameter/field. `None` (the default) reproduces every prior version's
exact behavior, byte-for-byte -- dense assembly, the existing MPC
function, and `femtoolkit.analysis.system.solve`. When a solver is
given, the stiffness/conductivity matrix is assembled in whichever
representation the solver's own `MATRIX_TYPE` requires, and
`analysis.last_solver_result` is populated with the full
`SolverResult` after `solve()` returns. `TransientThermalAnalysis` and
the nonlinear (Newton-Raphson) thermal/mechanical paths are unaffected
by this version -- see "Nonlinear and dynamic compatibility" below.

Verified end to end: a mechanical cantilever and a thermal
plate solved via dense, sparse-direct, and Conjugate Gradient agree to
within `1e-6`-`1e-8` (`tests/test_static_linear_solver_integration.py`,
`tests/test_thermal_solver_integration.py`).

## Nonlinear and dynamic compatibility

Version 26 does not redesign `NonlinearAnalysis` or `DynamicAnalysis`
(spec sections 19-20 explicitly exclude new nonlinear/dynamic
algorithms). The architecture does not prevent a future version from
using it there, though: a future Newton-Raphson iteration could call
`SparseDirectSolver().solve(tangent_system)` per iteration exactly as
the linear path does now, and `assemble_global_mass_sparse` (added
alongside the stiffness assembler, sharing the same COO-accumulation
helper) gives a future dynamic solver
(`M u'' + C u' + K u = F(t)`) a sparse mass matrix over the identical
DOF numbering as a sparse stiffness matrix, with zero further assembly
work needed.

## Solver selection

```python
from femtoolkit.solvers import create_solver

solver = create_solver(
    matrix_type="sparse",           # "dense" | "sparse"
    solver_type="conjugate_gradient",  # "direct" | "conjugate_gradient"
    tolerance=1e-8,
    max_iterations=1000,
)
```

`("dense", "conjugate_gradient")` is rejected outright
(`UnsupportedSolverError`): this architecture only offers CG over a
sparse representation. `femtoolkit.application.model_service.ModelService.build_solver`
maps a project's `SolverConfig` (`matrix_type`, `solver_type`,
`tolerance`, `max_iterations`) onto this same factory, returning `None`
for the default `("dense", "direct")` combination so the exact
zero-risk default path is preserved end to end from the GUI down to the
solver.

## GUI integration

The Solver page (`streamlit run src/femtoolkit/gui/app.py`) exposes
Matrix Type and Solver selectors (Conjugate Gradient only offered once
Sparse is selected), plus Tolerance/Maximum Iterations (enabled only
for Conjugate Gradient) -- calling only
`femtoolkit.application.validation.validate_solver` and
`ModelService.build_solver`; no solver logic lives in the Streamlit
page itself. After a run, the Run page displays "Analysis Complete"
with Solver/DOFs/Solve Time/Status metrics, non-zero-entry count and
density for a sparse solve, and iteration count/final residual for an
iterative one -- every value read directly from the `SolverResult` the
solver actually returned, never fabricated.

## Visualization

No new visualization code was added (spec section 25). The Results and
3D Visualization pages consume the same `SimulationResult` regardless
of which solver produced the underlying displacement/temperature array
-- verified directly: a project solved with Conjugate Gradient produces
identical Results-page numbers to the same project solved with the
default dense path.

## Example

```bash
python examples/solvers/sparse_solver_comparison.py
```

Assembles a real cantilever Q4 plate's stiffness matrix, compares dense
vs. sparse memory footprint and solve time across all three solvers,
and checks numerical agreement -- printing the run's actual numbers
rather than asserting a fixed ordering, per spec section 23's explicit
instruction not to claim sparse always wins.

## Limitations

- `TransientThermalAnalysis`, `NonlinearAnalysis`, and `DynamicAnalysis`
  remain dense-only in this version (see "Nonlinear and dynamic
  compatibility" above).
- Only one iterative method (Conjugate Gradient) is implemented; no
  preconditioning (see the Version 27 preview).
- `check_direct_solve_residual`'s threshold (`1e-4`) is a practical,
  empirically-verified cutoff for detecting an undetected-singular
  direct solve, not a rigorous condition-number bound.
- No GPU, distributed-memory (MPI), or external/commercial solver
  integration.

## Version 27 preview (not implemented)

**Advanced Solver Performance & Parallel Computation** -- improved
sparse assembly performance, solver profiling, parallel assembly,
parallel numerical operations, advanced preconditioners, scalable
iterative solvers, larger-model benchmarking, memory optimization, and
optional multiprocessing/threading strategies, aimed at improving
computational scalability on top of the sparse solver foundation this
version establishes.
