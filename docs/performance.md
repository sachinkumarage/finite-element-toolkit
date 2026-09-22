# Advanced Solver Performance & Parallel Computation (Version 27)

Performance profiling, a serial/parallel execution abstraction for
independent per-element work, a benchmarking framework, and post-run
performance diagnostics -- laid over the Version 26 sparse solver
infrastructure without changing any element formulation, material
model, or solver algorithm. This document is the detailed technical
guide; see the main [README](../README.md#version-27) for a shorter
overview and how Version 27 fits into the toolkit's history.

## Why performance work now

Version 26 made large models *solvable* (sparse storage, sparse direct
and iterative solvers). Version 27 does not touch that solving step at
all -- it profiles the stages *around* it (mesh preparation, element
computation, assembly, the solve call itself, post-processing) and
parallelizes the one stage that is safe to parallelize without rewriting
any solver: element-level computation.

## Computational cost in FEA

For a mesh with `N_e` elements, each element's local matrix is computed
independently:

$$
K_e = f(\text{element geometry}, \text{material}, \text{state})
$$

Nothing about element `i`'s stiffness matrix reads or writes element
`j`'s data -- this is what makes element computation **embarrassingly
parallel**. Global assembly then combines every element's contribution:

$$
K = \sum_e A_e^T K_e A_e
$$

where `A_e` is the element-to-global DOF mapping (in this toolkit,
`dof_map.global_index(node_id, dof)` per `(node_id, dof)` pair in the
element's `dof_keys()`). This step is **not** embarrassingly parallel in
the same way: several elements can share a global DOF (two elements
meeting at a node), so naively writing several elements' contributions
into the same global matrix at once from different workers is a race
condition. Assembly stays sequential in this toolkit (it is already
fast -- `O(nnz)` for the sparse assembler, Version 26) and only *element
computation* -- the actually expensive, embarrassingly parallel part --
is what Version 27 makes parallelizable.

| Stage | Parallelizable? | Why |
|---|---|---|
| Mesh preparation | Not addressed this version | One-time, typically cheap relative to solving |
| Element computation | **Yes** (Version 27) | Each element's matrix depends only on its own data |
| Global assembly | No (kept sequential) | Shared-DOF writes are a race condition; already fast |
| Linear solve | No (kept sequential, Version 27 scope) | See "Solver parallelism" below |
| Post-processing | Yes in principle, not wired in this version | Per-element stress/strain recovery is also independent; see Limitations |

## The execution abstraction

```text
ElementExecutor (ABC)
|-- SerialExecutor    -- default, byte-for-byte the prior versions' loop
`-- ParallelExecutor  -- opt-in, ProcessPoolExecutor or ThreadPoolExecutor
```

Both implement one method, `map(func, items) -> list`, always returning
results **in the same order as `items`**, regardless of which worker
finishes first -- `concurrent.futures.Executor.map` guarantees this
internally, and both executors preserve it. A caller never needs to
track which element ID a result belongs to by completion order.

```python
from femtoolkit.execution import ExecutionConfig, create_executor

executor = create_executor(ExecutionConfig(mode="parallel", workers=4))
results = executor.map(some_function, items)
```

`ExecutionConfig` is immutable and validated (`InvalidExecutionConfigurationError`
on a non-positive `workers`/`chunk_size` or an unknown `mode`/`backend`).
Its default, `ExecutionConfig()` (`mode="serial"`), is what every
analysis uses when no `execution` argument is given at all --
`StaticLinearAnalysis`/`SteadyStateThermalAnalysis`'s `execution: ExecutionConfig
| None = None` parameter reproduces every prior version's exact serial
behavior when left at its default, exactly mirroring how Version 26's
`solver` parameter worked.

### Process vs. thread backend

- **`backend="process"`** (the default) -- true parallelism via
  `concurrent.futures.ProcessPoolExecutor`, bypassing Python's GIL
  entirely. Costs: every task's arguments and result must be pickled to
  cross the process boundary, and `func` must be a plain, importable,
  module-level callable (not a lambda, a closure, or a bound method of
  an unpicklable object) -- see `TaskSerializationError`.
- **`backend="thread"`** -- `concurrent.futures.ThreadPoolExecutor`, no
  pickling cost and no picklability restriction, but limited by the GIL
  except during the C-level NumPy/SciPy calls an element's own matrix
  computation releases it for.

Neither is unconditionally faster -- `examples/performance/serial_vs_parallel.py`
measures both against serial on the same mesh.

## Parallel element computation in this toolkit

`femtoolkit.analysis.parallel_assembly` supplies the "compute every
element's contribution" step through an `ElementExecutor`:

```python
from femtoolkit.analysis.parallel_assembly import compute_stiffness_contributions
from femtoolkit.execution import ExecutionConfig, create_executor

executor = create_executor(ExecutionConfig(mode="parallel", workers=4))
contributions = compute_stiffness_contributions(mesh.elements, executor)
# feeds the *unchanged* Version 2/26 assembler:
stiffness = assemble_global_stiffness(dof_map, contributions)
```

`compute_conductivity_contributions` is the thermal analogue, packing
`(element, material, evaluation_temperature)` into one argument per task
(an `ElementExecutor` only maps single-argument callables). Both are
verified to produce **exactly** (not just approximately) the same result
as the equivalent serial loop, for a real Q4 mesh and a real thermal
plate (`tests/test_parallel_assembly.py`) -- the underlying computation
is identical; only where it runs differs.

### Integration: `StaticLinearAnalysis` / `SteadyStateThermalAnalysis`

Both gained an optional `execution: ExecutionConfig | None = None`
parameter (mirroring Version 26's `solver` parameter exactly) and a
`last_performance_report: PerformanceReport | None` attribute, populated
after every `solve()` call. `SteadyStateThermalAnalysis`'s nonlinear
Newton-Raphson path (temperature-dependent convection or radiation) is
**unaffected** by `execution` -- it always uses its existing serial,
dense code path, exactly like it was already unaffected by `solver` in
Version 26 -- `last_performance_report` stays `None` on that path.
`TransientThermalAnalysis`, `NonlinearAnalysis`, and `DynamicAnalysis`
were not touched.

## Sparse assembly strategy: unchanged, deliberately

Version 27 does **not** modify `femtoolkit.analysis.sparse_assembly`.
The COO-triplet-accumulation strategy that module already uses (Version
26) is itself embarrassingly parallel *in principle* (each element's
triplets are independent), but assembling triplets is already `O(nnz)`
and fast in practice (microseconds to low milliseconds for meshes in
the thousands of elements) -- profiling evidence did not justify adding
parallel complexity to an already-cheap step (spec section 17: "only
optimize operations supported by profiling evidence"). What dominates
is element *computation*, not triplet accumulation, so that is what this
version parallelizes.

## Performance profiling

`femtoolkit.performance.Profiler` measures named stages with
`time.perf_counter` (`MESH`, `ELEMENT`, `ASSEMBLY`, `BOUNDARY_CONDITION`,
`SOLVE`, `POST_PROCESSING`):

```python
from femtoolkit.performance import Profiler, ELEMENT, ASSEMBLY

profiler = Profiler()
with profiler:
    with profiler.stage(ELEMENT):
        ...
    with profiler.stage(ASSEMBLY):
        ...
report = profiler.report(model_name="Cantilever", node_count=..., element_count=...)
```

`PerformanceReport` never fabricates a number for a stage that was not
measured (a `None` field, not `0.0`) -- `StaticLinearAnalysis`/
`SteadyStateThermalAnalysis` leave `boundary_condition_time` and
`post_processing_time` as `None`: this toolkit's architecture fuses
boundary-condition elimination into the solve call itself (see
`femtoolkit.solvers.base`'s `partition_dofs`/`reduced_system`), so there
is no separately-measurable BC-processing stage to report, and neither
analysis does post-processing inside `solve()` (stress/strain recovery
happens afterward, on `AnalysisResult`). Reporting a fabricated `0.0`
for either would misleadingly imply "measured, took no time" rather
than "not separately measurable here."

`PerformanceReport.solver_result` carries the full Version 26
`SolverResult` (including its `diagnostics` dict's `dofs`/`nnz`/
`density`) rather than duplicating those fields -- extending Version
26's diagnostics, not reinventing them (spec section 5).

## Batch processing and memory

`ExecutionConfig.chunk_size` (`"auto"` by default) controls how many
elements are handed to a worker per task batch.
`resolve_chunk_size` targets roughly four chunks per worker --
enough for `concurrent.futures` to load-balance uneven per-element cost
across workers without dispatching one task per element (which would
spend most of the parallel budget on inter-process communication for a
typical, cheap FEA element). This toolkit's element matrices are small
enough (a handful of floats to a few hundred, depending on element type)
that Version 27 does not implement a separate streaming/batched
*assembly* path -- `compute_stiffness_contributions` still returns a
full list of contributions, matching every prior version's data flow.
The chunking that exists is purely a worker-dispatch granularity
control, not a memory-management feature; see Limitations for what a
genuinely memory-streamed pipeline would need.

## Speedup, efficiency, and Amdahl's Law

$$
S_p = \frac{T_1}{T_p} \qquad E_p = \frac{S_p}{p}
$$

where `T_1` is serial time, `T_p` is parallel time with `p` workers.
`E_p = 1.0` is ideal linear scaling; real efficiency is lower because of
serial portions, synchronization, communication overhead, memory
bandwidth, and process-startup cost. Amdahl's Law quantifies the ceiling
this imposes:

$$
S(N) = \frac{1}{(1 - P) + \frac{P}{N}}
$$

where `P` is the parallelizable fraction of the total work and `N` is
the worker count. For this toolkit's simple, closed-form 2D element
formulations, per-element computation is a genuinely small fraction of
total analysis time next to assembly and solving at most model sizes --
so `P` is small, and Amdahl's Law predicts (correctly, per
`examples/performance/scaling_study.py`'s measurements) that no amount
of parallelizing *element computation alone* delivers large overall
speedup. This is exactly why Version 27 does not attempt to parallelize
the solve itself (see below) and why the GUI/documentation never claims
guaranteed speedup.

## Solver parallelism: explicitly out of scope

Version 27 does not implement a custom parallel sparse solver.
`SparseDirectSolver`/`ConjugateGradientSolver` (Version 26) are
unmodified; if SciPy's underlying LAPACK/BLAS/SuperLU backend uses
internal threading, this toolkit does not interfere with it or attempt
to layer its own parallelism on top (which would risk oversubscription
-- more total threads/processes than physical cores, thrashing rather
than speeding up). `PerformanceReport`/`SolverResult` expose solver
timing so a user can *observe* solver cost, but Version 27 deliberately
leaves solver-internal parallelism as a Version 28 concern (see the
preview below).

## Worker configuration

```python
ExecutionConfig(
    mode="parallel",       # "serial" (default) or "parallel"
    workers=4,              # None = automatic, min(os.cpu_count(), 8)
    chunk_size="auto",      # "auto" or a positive integer
    backend="process",      # "process" (default) or "thread"
)
```

The automatic worker default is capped at 8 regardless of core count
(`resolve_workers`) -- using every available core by default would
starve the rest of the user's machine and, for the small-to-medium
meshes this toolkit is typically used for, adds process-startup
overhead disproportionate to the benefit; a user who genuinely wants
more can pass `workers` explicitly. `InvalidExecutionConfigurationError`
rejects zero/negative workers, zero/negative chunk sizes, and unknown
mode/backend names before any worker is spawned.

## Worker lifecycle and error handling

A fresh `ProcessPoolExecutor`/`ThreadPoolExecutor` is created for each
`ParallelExecutor.map()` call and cleanly shut down (`with pool: ...`)
before returning -- no persistent background pool is kept alive between
calls (spec section 14: "do not create persistent background workers
unless genuinely necessary"). An empty workload (`items == []`) returns
`[]` immediately without starting any workers. Two dedicated exceptions
distinguish worker-boundary failures from ordinary computation errors:

- `TaskSerializationError` -- a `func`/item could not be pickled to send
  to a worker process (e.g. a lambda passed with `backend="process"`).
- `WorkerExecutionError` -- a worker raised while executing a task; the
  original exception is chained via `__cause__` so its type and message
  remain inspectable.

Neither hides the failure or silently falls back to a wrong answer.

## Deterministic results

Parallel and serial element computation are verified to produce
**exactly** (`max |difference| == 0.0`, not just within a tolerance) the
same global stiffness/conductivity matrix and the same solved
displacement/temperature field, for both the process and thread
backends (`tests/test_parallel_assembly.py`,
`tests/test_static_linear_execution_integration.py`,
`tests/test_thermal_execution_integration.py`). This is expected
here: unlike, say, a parallel reduction that sums floating-point values
in a different order per run, this toolkit's parallel path computes
each element's matrix independently and combines them through the
*same*, sequential, unchanged assembler every time -- there is no
floating-point-order source of non-determinism to tolerate. A project
combining this pattern with a genuinely parallel *reduction* (which
Version 27 does not do) would need to document a tolerance for tiny
floating-point accumulation differences instead.

## GUI integration

The Solver page (Version 24/26) gained an **Execution Settings**
section: Mode (Serial/Parallel), Workers (with an "Automatic" checkbox),
and Batch Size, validated via `validate_execution`. The Run page's
post-run summary gained a **Simulation Performance** section (element,
assembly, and solve time; execution mode and worker count; total time)
displayed *alongside*, not replacing, the existing solver-diagnostics
summary -- performance information is secondary to the engineering
results, matching spec section 26's explicit instruction not to turn the
GUI into a benchmarking dashboard. Result visualization (Version 22/23)
is unaffected regardless of execution mode, exactly like it was
unaffected by solver choice in Version 26.

## Reproducibility and platform considerations

Every benchmark example prints the numbers that specific run actually
produced, with an explicit note that hardware, core count, current
system load, and even background OS activity change the results --
`examples/performance/scaling_study.py`'s numbers on one machine are not
a claim about any other machine. `ExecutionConfig`/`ParallelExecutor`
use only the Python standard library (`concurrent.futures`,
`multiprocessing` underneath it) and make no platform-specific
assumptions; `ProcessPoolExecutor` behaves consistently across macOS,
Linux, and Windows for the plain, module-level, picklable functions this
toolkit passes to it (macOS/Windows default to the `spawn` start method,
Linux to `fork` -- both work correctly here since every task function
lives in an importable module, never a closure or `__main__`-local
function).

## Limitations

Version 27 does **not** include: a custom parallel sparse solver (see
"Solver parallelism" above); parallel post-processing (per-element
stress/strain recovery is architecturally just as independent as
stiffness computation, but was not wired into `ElementExecutor` this
version -- profiling did not show it as a bottleneck relative to
assembly/solving for the models exercised); a genuinely
memory-streamed/batched assembly pipeline (chunking here controls
worker-dispatch granularity, not peak memory -- `compute_stiffness_contributions`
still materializes a full contribution list); advanced preconditioners
or GMRES (Version 28 preview, below); GPU computing, CUDA, MPI/distributed-memory
or cluster/cloud computing; automatic tuning of `workers`/`chunk_size`
beyond the fixed automatic-worker cap. `TransientThermalAnalysis`,
`NonlinearAnalysis`, and `DynamicAnalysis` remain entirely unaffected by
this version's execution infrastructure.

## Version 28 preview (not implemented)

**Advanced Preconditioning & Scalable Iterative Solvers** -- Jacobi and
incomplete-LU/incomplete-Cholesky preconditioning, improved Conjugate
Gradient and GMRES workflows, preconditioner selection, richer solver
convergence diagnostics, more robust handling of difficult (ill-conditioned)
sparse systems, iterative-solver benchmarking, and improved large-model
robustness -- building on the execution and performance infrastructure
Versions 26-27 establish.
