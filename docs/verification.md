# FEA Verification, Validation & Engineering Reporting (Version 29)

A professional framework for systematically answering "can I trust this
simulation result?" -- analytical benchmarks, error metrics, tolerance
handling, mesh convergence studies, solver convergence verification,
equilibrium checks, a validation-data comparison framework, reproducibility
metadata, and engineering report generation, laid over the toolkit's
existing element formulations and solvers without changing any of them.
This document is the detailed technical guide (verification, plus the
developer-facing architecture); see [`docs/validation.md`](validation.md)
for the validation guide, [`docs/reporting.md`](reporting.md) for the
reporting guide, and the main [README](../README.md#version-29) for a
shorter overview.

## Verification vs. validation

These two words are used precisely and consistently throughout this
toolkit's API, never interchangeably:

- **Verification** asks *"are we solving the equations correctly?"* --
  a purely mathematical/numerical question, answered by comparing a
  numerical result against a *known* reference: a closed-form
  analytical solution, a mesh-refinement trend, agreement between two
  different solvers on the same problem, or a first-principles
  conservation law (equilibrium). No external data is required --
  `femtoolkit.verification` needs nothing but the toolkit itself.
- **Validation** asks *"are we solving the correct physical problem?"*
  -- a question that can only be answered against *external, trusted
  evidence*: experimental measurements, published reference data, or a
  prior high-fidelity simulation. `femtoolkit.validation` provides the
  *infrastructure* for this comparison but ships with **no bundled
  dataset** and performs **no validation on its own** -- a model this
  toolkit has not been given real reference data for is *unverified for
  physical correctness*, full stop, never silently assumed valid.

Every result, report section, and GUI panel in Version 29 keeps these
two words in their own lane -- a verification result never claims
validation, and a report's "Validation Results" section says plainly
that no comparison was made when no dataset was supplied (see
`render_markdown`'s exact wording in
[`docs/reporting.md`](reporting.md)).

## Structured status vocabulary

Every comparison in this framework resolves to one of five statuses
(`femtoolkit.verification.status.VerificationStatus`), never a bare
boolean or a printed message:

| Status | Meaning |
|---|---|
| `PASS` | The numerical result is within tolerance of the reference. |
| `FAIL` | The numerical result exceeds tolerance. |
| `WARNING` | A result exists but calls for engineering review (e.g. a mesh convergence study that has not yet settled within the tested levels). |
| `NOT_AVAILABLE` | The comparison could not be made -- required reference data or an executable case is unavailable. Never silently treated as a pass. |
| `NOT_RUN` | The case has been defined but not yet executed. |

## The verification workflow

```text
VerificationCase                 (femtoolkit.verification.cases)
       |
       v
  case.run() -- builds + solves an FEA model, extracts a quantity
       |
       v
  compare numerical vs. reference (femtoolkit.verification.metrics)
       |
       v
  apply tolerance                (femtoolkit.verification.tolerance)
       |
       v
  VerificationResult              (femtoolkit.verification.cases)
```

`VerificationRunner.run(case)` executes exactly this pipeline for one
case, catching any exception a case's own `run()` raises and degrading
to `NOT_AVAILABLE` rather than aborting an entire run over one bad case
(spec section 22: "handle unsupported verification cases gracefully").
`VerificationRunner.run_all(cases)` collects a `VerificationReport` with
summary counts (`passed`, `failed`, `warnings`, `not_available`,
`all_passed`).

## Error metrics

`femtoolkit.verification.metrics` implements exactly the handful of
metrics classical FEA verification needs, each a small, pure function:

```text
absolute_error      e_abs  = |x_fea - x_ref|
relative_error       e_rel  = |x_fea - x_ref| / max(|x_ref|, eps)
l2_error             ||e||_2 = sqrt(sum((x_fea_i - x_ref_i)^2))
relative_l2_error             = ||x_fea - x_ref||_2 / max(||x_ref||_2, eps)
energy_norm_error              = sqrt(e^T K e) / max(sqrt(u_ref^T K u_ref), eps)
```

`energy_norm_error` is the standard norm for assessing displacement-based
FEA convergence: the strain energy a displacement *error* vector would
store in the structure, using the same stiffness matrix that assembled
the analysis -- physically meaningful in a way a plain L2 norm over
mixed-direction DOF components is not. Every relative metric takes a
configurable `epsilon` floor (`DEFAULT_EPSILON = 1e-12`) for its
denominator, avoiding division by (near-)zero when the reference value
is itself near zero (a reaction at a symmetric load point, a heat
source in a purely-conductive model).

## Tolerance handling

`femtoolkit.verification.tolerance.Tolerance` implements the standard
combined absolute/relative criterion (matching `numpy.isclose`'s own
formula) rather than a single fixed tolerance or floating-point exact
equality:

```text
|numerical - reference| <= absolute_tolerance + relative_tolerance * |reference|
```

A combined tolerance matters in practice, not just in principle: a
purely relative criterion is meaningless when the reference value is
itself near zero (see the thermal energy balance discussion below,
where this was a genuine bug caught during development), and a purely
absolute criterion does not scale sensibly across problems of very
different magnitude. `Tolerance.__post_init__` rejects a tolerance where
both components are exactly zero -- that degenerates to exact equality,
which this framework never uses.

## Analytical benchmark suite

`femtoolkit.verification.benchmarks` builds real FEA models (using the
toolkit's existing, unmodified elements and analyses) and compares them
against closed-form solutions computed from the same input parameters
-- no benchmark result is ever hard-coded:

| Benchmark | Element(s) | Reference formula |
|---|---|---|
| Axial bar | `BarElement` | `delta = F*L/(A*E)`, `sigma = F/A` |
| Two-bar truss | `TrussElement2D` x2 | `N = -F*L/(2*h)`, `delta = F*L^3/(2*h^2*A*E)` (derived by joint equilibrium + unit-load method) |
| Cantilever beam | `FrameElement2D` | `delta = -F*L^3/(3*E*I)`, `M = F*L` (Euler-Bernoulli) |
| 1D thermal conduction | 1D conduction elements | `T(x) = T0 + (TL-T0)*x/L`, `q = -k*(TL-T0)/L` |
| Q4 / CST / HEX8 patch tests | `QuadElement2D` / `CSTElement2D` / `Hex8Element3D` | Exact constant-strain reproduction of a prescribed linear displacement field |

Every element formulation above is *exact* for the field it represents
-- a bar element for uniform axial strain, a frame element for the
Euler-Bernoulli beam equation, Q4/CST/HEX8 for a linear displacement
field -- so each benchmark's default tolerance
(`Tolerance(absolute=1e-9, relative=1e-6)`) is deliberately tight: the
FEA result should match theory to floating-point precision, not merely
"be close." `standard_benchmark_suite()` returns every case with its
default parameters. `hex8_patch_case` is this version's **3D solid
benchmark** (spec section 4.6): the same constant-strain patch test
already proven stable in `tests/validation/test_hex8_patch.py` since
Version 15, packaged as a `VerificationCase`.

## Mesh convergence studies

`femtoolkit.verification.convergence.run_mesh_convergence_study` runs
several mesh resolutions and records the relative change in a tracked
result quantity between consecutive levels:

```text
Delta_i = |Q_i - Q_{i-1}| / max(|Q_i|, eps)
```

**Convergence is not assumed to be monotonic.** A result can
legitimately oscillate while settling toward a stable value (common
near a stress concentration with a coarse mesh); the study records the
full history rather than asserting a monotonically decreasing sequence.
The study's overall `status` is `PASS` if the *final* level's relative
change is within `tolerance`, `WARNING` if not (the model is not judged
wrong -- only that convergence was not demonstrated within the tested
levels), and `NOT_AVAILABLE` with fewer than two levels (a relative
change needs a predecessor).

This is explicitly distinct from **solver convergence** (an iterative
*linear solve*'s residual reaching its tolerance within a fixed mesh)
-- the next section.

## Solver convergence verification

`femtoolkit.verification.solver_verification` extends the Version 26
solver diagnostics (`SolverResult`) rather than re-deriving them:
`solver_convergence_record(solver_result)` packages `solver_name`,
`matrix_type` (inferred from whether sparse-specific diagnostics are
present), `iterations`, `final_residual`, `relative_residual`,
`converged`, and `solve_time` into a `SolverConvergenceRecord`.
`preconditioner` is always `None` -- this toolkit does not yet
implement solver preconditioning (see the Version 28 preview in
[`docs/performance.md`](performance.md)) -- reported honestly as
unavailable, never fabricated.

`compare_solver_solutions(name, reference, comparison, tolerance)`
compares two solvers' solution vectors for the same problem (e.g. a
dense direct solve against a Conjugate Gradient solve) using the same
`VerificationResult` shape every other comparison in this framework
uses.

### Residual history (opt-in)

`ConjugateGradientSolver` gained an opt-in `track_residual_history: bool
= False` field (Version 29, additive to the Version 26 solver -- default
`False` costs every existing caller nothing). When enabled, the solver's
existing per-iteration callback also computes the relative residual and
appends it to `diagnostics["residual_history"]`. This is disabled by
default because computing a residual at every iteration costs one extra
sparse matrix-vector product per iteration, roughly doubling a solve's
cost -- a real cost only paid when a caller explicitly wants the
history, e.g. for `femtoolkit.verification.plots.plot_residual_history`.

## Equilibrium checks

`femtoolkit.verification.checks` verifies a conservation law the
underlying formulation guarantees at its own converged solution --
catching a modeling error (wrong boundary condition, a load that never
reached the structure) that a numerically converged solve would not
otherwise reveal:

- `check_force_equilibrium(dof_map, forces, reactions, tolerance)` --
  groups every global DOF by physical component using
  `DOFMap`'s node-major layout (`global_index = node_position *
  dofs_per_node + local_dof`), so `forces[c::dofs_per_node]` and
  `reactions[c::dofs_per_node]` are exactly every DOF acting in
  direction `c` (X, Y, RZ, ...). Checks `reaction_total ~= -applied_total`
  per component using the combined `Tolerance` (so the relative term
  scales with the actual applied-load magnitude in that direction,
  rather than being compared against a fixed reference of zero).
- `check_thermal_energy_balance(analysis, result, tolerance)` --
  reuses the existing Version 21
  `femtoolkit.thermal.thermal_analysis.steady_state_energy_balance`
  directly (`q_supplied` vs. `q_removed`, computed through two
  independent paths) rather than re-deriving it, and compares them
  with a combined `Tolerance`.

**A real bug this combined-tolerance design caught during development:**
a model with only prescribed-temperature boundaries and no explicit
heat source has `q_supplied == 0` exactly. A naive relative-only check
(`|q_supplied - q_removed| / max(|q_supplied|, eps)`) divides a
physically negligible floating-point mismatch (~1e-13 W) by the
`epsilon` floor itself, producing a spuriously large "relative failure."
Switching to `Tolerance.is_satisfied(q_removed, q_supplied)` (an
absolute term that does not vanish when the reference is zero) fixed
this -- verified directly in `tests/verification/test_checks.py::test_thermal_energy_balance_near_zero_supply_does_not_spuriously_fail`.

`SimulationService` (Version 24, extended Version 29) now computes an
equilibrium check automatically for every completed run --
`SimulationRunResult.equilibrium_check` -- so the GUI can display it
without performing any checking itself.

## Verification plots

`femtoolkit.verification.plots` provides four thin adapters over the
existing Matplotlib plotting primitives
(`femtoolkit.postprocessing.visualization`, Version 22, extended this
version with `plot_semilog_line`/`plot_comparison_line`): no plotting
logic lives in the verification package itself, only the conversion
from a verification data structure into the `x`/`y` arrays those
functions expect.

- `plot_mesh_convergence(study)` -- result quantity vs. mesh size.
- `plot_error_vs_mesh_size(study)` -- relative change vs. mesh size (log
  scale).
- `plot_residual_history(residual_history, solver_name)` -- solver
  relative residual vs. iteration (log scale).
- `plot_numerical_vs_reference(x, numerical, reference, quantity)` --
  both series on one set of axes.

## Developer documentation: architecture and extension points

```text
src/femtoolkit/
    verification/
        status.py               VerificationStatus enum
        metrics.py               error metrics (pure functions)
        tolerance.py              Tolerance (combined absolute/relative)
        cases.py                  VerificationCase, VerificationResult
        runner.py                 VerificationRunner, VerificationReport
        benchmarks.py             analytical benchmark case builders
        convergence.py            mesh convergence study framework
        checks.py                 equilibrium checks
        solver_verification.py    solver convergence + comparison
        plots.py                  verification-specific plot adapters
    validation/                  see docs/validation.md
    reporting/                   see docs/reporting.md
```

**Adding a new analytical benchmark.** Write a function in
`benchmarks.py` (or your own module) returning one or more
`VerificationCase` instances: state the reference formula in the
docstring, compute `reference_value` from the function's own
parameters (never hard-code a number), and give `run` a closure that
builds the mesh/analysis, solves, and returns the matching numerical
quantity. `VerificationRunner` needs nothing case-type-specific.

**Adding a new error metric.** Add a pure function to `metrics.py`
following the existing signature convention (`(numerical, reference,
epsilon=DEFAULT_EPSILON) -> float`) -- `VerificationRunner` already
dispatches on whether a case's reference value is scalar or vector
(`_is_vector`), so a new *scalar* metric needs no runner changes; a new
*vector* metric would need one dispatch branch added to
`VerificationRunner.run`.

**Adding a new equilibrium check.** Follow `checks.py`'s pattern:
return an `EquilibriumCheckResult` (reuse `EquilibriumComponent` if the
check naturally decomposes by direction/DOF, otherwise leave
`components` empty like the thermal check does), compute a genuine
physical quantity from an existing, unmodified analysis/result object,
and use a `Tolerance` (never a bare relative fraction) for the
pass/fail decision.

## Limitations

Version 29 does not include: advanced solver preconditioning (Version
28 scope, not yet implemented -- `SolverConvergenceRecord.preconditioner`
is always `None`); a parallel/distributed verification-case runner
(`VerificationRunner` runs cases serially; nothing prevents wiring
Version 27's `ElementExecutor` in later, but this version does not);
automatic mesh generation *for* a convergence study beyond what the
caller supplies (each `MeshConvergenceLevel` is a plain callable the
caller writes); a PDF report renderer (see
[`docs/reporting.md`](reporting.md) for why); and any bundled
experimental or published reference dataset (see
[`docs/validation.md`](validation.md)).
