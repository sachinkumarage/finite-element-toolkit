# Advanced Parallel Simulation Execution & High-Performance Study Orchestration (Version 34)

A local, process-based execution layer for running **independent whole
simulation tasks** in parallel -- a parameter study's scenarios, a Monte
Carlo study's samples, an optimization's candidate designs -- laid over
the existing Version 24 solve pipeline without changing a single FEA
algorithm, material model, or solver. This document is the detailed
technical guide; see the main [README](../README.md#version-34) for a
shorter overview and how Version 34 fits into the toolkit's history.

## Why this is a different layer from Version 27

Version 27 (`femtoolkit.execution`, see [`performance.md`](performance.md))
already parallelizes *independent elements' stiffness-matrix
computation within one FEA assembly* -- fine-grained, intra-solve
parallelism, wired into `femtoolkit.analysis.parallel_assembly` and
exposed through `Project.execution`. Version 34
(`femtoolkit.orchestration`) is a completely different, coarser-grained
layer one level up: it parallelizes *entire, independent simulation
runs*, each of which may itself still use Version 27's element-level
parallelism internally.

```text
Simulation Study / Monte Carlo Study / Optimization Batch
                        |
                 Task Generation
                        |
                ExecutionManager            femtoolkit.orchestration.manager
                        |
                ExecutionBackend             backends/: serial, local_process
                        |
                  Worker Pool                local_process only
                        |
            SimulationRunner / Evaluator     reused unchanged
                        |
                  TaskOutcome / SimulationRun
```

The two layers' configuration objects, exceptions, and module names are
kept entirely distinct (`OrchestrationConfig` vs. `ExecutionConfig`,
`ExecutionBackend` here vs. `ElementExecutor` there) so neither is ever
confused with the other, and `femtoolkit.execution` is untouched by this
version.

## Serial vs. parallel execution

Every integration point (`StudyRunner.run`, `MonteCarloRunner.run`,
`OptimizationAlgorithm.optimize`, `OptimizationRunner.run`) takes an
optional `orchestration_config: OrchestrationConfig | None` parameter.
`None` (the default everywhere) reproduces every prior version's exact
sequential behavior -- nothing changes unless a caller opts in.

```python
from femtoolkit.orchestration import OrchestrationConfig
from femtoolkit.studies.runner import StudyRunner

# Serial -- identical to Version 30's original behavior.
result = StudyRunner().run(study)

# Parallel -- the same study, spread across local worker processes.
config = OrchestrationConfig(execution_mode="parallel", max_workers=4)
result = StudyRunner().run(study, orchestration_config=config)

# Parallel, with the batch's ExecutionSummary (completed/failed/cancelled
# counts, elapsed time, speedup/efficiency where meaningful):
result, summary = StudyRunner().run_with_summary(study, orchestration_config=config)
```

`MonteCarloRunner.run(config, orchestration_config=...)` and
`OptimizationRunner.run(problem, config, orchestration_config=...)`
follow the identical pattern.

## Worker processes, not threads

`LocalProcessBackend` uses `concurrent.futures.ProcessPoolExecutor`, not
a thread pool. Each task here runs real Python-level control flow
(building a scenario, orchestrating a solve) interleaved with
NumPy/SciPy calls; CPython's GIL means a thread pool would only gain
real parallelism during the portions that release it. A process pool
sidesteps the GIL entirely, at the cost of requiring every task's
function, arguments, and result to be **picklable** to cross the
process boundary.

**Picklability.** A task's function must be a plain module-level
function (`femtoolkit.orchestration.simulation.execute_simulation_task`,
`femtoolkit.optimization.batch.execute_design_evaluation_task`), never a
lambda or a nested closure. This mattered for the optimization layer in
particular: `from_result_extractor`, `robust_objective_mean`,
`robust_objective_statistic`, and `robust_constraint_statistic` all
return small, frozen `@dataclass` callables (not closures) specifically
so an `Objective`/`Constraint` built from them can cross a process
boundary -- see `femtoolkit.optimization.objectives`'s module docstring.
A caller-supplied `lambda` passed directly as an objective's `evaluate`
remains valid for serial execution, but raises `TaskSerializationError`
if parallel evaluation is requested for it; this is an inherent property
of Python multiprocessing, not a toolkit defect.

**A fresh pool per batch.** Exactly like Version 27's `ParallelExecutor`,
a new worker pool is created for each batch and always shut down before
the call returns -- never left running after cancellation or a timeout.
This means a batch split across many small sub-batches (for example, a
population-based optimization algorithm that batch-evaluates each
generation separately) pays pool-startup overhead once per sub-batch;
see "Recommended worker counts" below.

## Resource management

`max_workers=None` (the default) resolves to `min(os.cpu_count() or 1,
8)` -- never `os.cpu_count()` directly, since using every available core
by default can starve the rest of the machine. A caller who genuinely
wants more passes `max_workers` explicitly.

## Reproducibility and seed handling

**The hard requirement this version satisfies:** changing the number of
workers must never change a Monte Carlo study's generated samples.
Version 30's parameter sweeps and Version 31's Monte Carlo sampling both
already draw their entire sample/scenario set up front, in the calling
process, before any task is built or executed. This means the
requirement holds *structurally* -- execution order and worker count
can never affect a sample set that was fully determined before execution
began -- with zero new seed-derivation machinery needed for sampling
itself.

`femtoolkit.orchestration.random_state.derive_task_seed(base_seed,
task_id)` exists for the narrower, genuinely execution-order-sensitive
case: a task's own internal computation that needs reproducible
randomness of its own. It turns one base seed plus a task's stable
identifier into a deterministic child seed via `hashlib.sha256` (not
Python's salted built-in `hash()`), so the same `(base_seed, task_id)`
pair always derives the same seed regardless of execution order, worker
count, or serial vs. parallel execution.

**Optimization randomness stays separate from simulation randomness.**
A population-based algorithm's own RNG (selection, crossover, mutation)
never changes when its candidate evaluation is batched in parallel --
see "Population-based algorithms" below for exactly which generations
are batch-evaluated and why the un-batched ones are not.

## Population-based algorithms

`DifferentialEvolution`, `GeneticAlgorithm`, `ParticleSwarmOptimization`,
and `NSGA2` all batch-evaluate generation 0's initial population when
`orchestration_config` requests parallel execution -- always safe, since
every one of these algorithms' initial populations is independent by
construction.

`GeneticAlgorithm` and `NSGA2` additionally batch-evaluate each full
generation's offspring, because both algorithms' selection operators
(`_tournament_select`, `_crowded_tournament`) only ever read the
*frozen* parent generation -- never a value from a sibling child still
being produced in the same generation -- so a whole generation's
children can be produced (consuming RNG) independently of evaluation,
then evaluated as one batch.

`DifferentialEvolution` and `ParticleSwarmOptimization` do **not** batch
their per-generation loop: both are, by design, "steady-state"/
"asynchronous" variants where a later individual's update can see an
earlier individual's just-computed result within the same generation --
a real, intentional property of the implemented variant, not a bug.
Changing this to a synchronous/frozen-snapshot variant to make it
batchable would be a genuine algorithmic change this version does not
make (see each algorithm's own module docstring).

**A documented trade-off for GA/NSGA2's batched offspring.** Stopping
conditions that can fire *mid-generation* in the un-batched loop
(`max_consecutive_failures`, `CONVERGED`, `TARGET_REACHED`) are only
re-checked once per generation in batched mode, so up to one
generation's worth of extra evaluations may run past where serial
evaluation would have stopped early. `max_evaluations` itself is never
exceeded -- each batch is explicitly capped at the remaining budget.

## Failure handling

An individual task's failure never aborts the batch unless
`fail_fast=True`. `SimulationRunManager.execute()` itself never raises
for an ordinary solver/validation failure -- it returns a structured
`FAILED` `SimulationRun` -- so a batch's `ExecutionSummary.completed_tasks`
counts *tasks whose function returned without raising*, which is not the
same question as *how many simulations succeeded*; check each run's own
`RunStatus` for that. See `examples/orchestration/failure_handling.py`
for a worked example of this exact distinction.

`fail_fast=True` stops submitting new tasks after the first failure and
cancels any task still queued (never dispatched), but a task already
running in a worker when the failure is observed is allowed to finish --
a deliberately softer stop than forcibly killing in-flight work.

## Cancellation

`CancellationToken` is the cooperative mechanism every backend checks
between task submissions. Once cancelled, the worker pool is shut down
immediately (`cancel_futures=True`) and every task not already in a
terminal state is marked `CANCELLED` without waiting for it -- no
orphaned worker processes are left running. Cancellation does not
guarantee an already-running task's underlying OS process is killed
immediately; it guarantees the *batch* stops waiting for it and the pool
is torn down.

## Timeout, approximated as "no progress"

`OrchestrationConfig.timeout` bounds how long a batch may go *without
any task completing* -- not a true per-task wall-clock budget measured
from that task's own start. If an entire `timeout`-second window passes
with zero completions, every outstanding task is treated as timed out,
the pool is shut down, and no further tasks are submitted. This is a
conservative, deliberate approximation: a true per-task timeout would
need extra IPC machinery this version does not add.

## Nested-parallelism protection

Parallel optimization evaluates candidates in worker processes; a
robust (uncertainty-aware) objective evaluated inside one of those
workers runs its own inner Monte Carlo study; if that study also tried
to parallelize across its own worker pool, each outer worker would spawn
a further pool of its own -- `workers x workers` processes for what the
caller asked to run on `workers` processes.

`femtoolkit.orchestration.nesting.is_inside_worker_process()` answers
"is the *current* process itself a pool worker" via
`multiprocessing.parent_process()`, reliably and portably, regardless of
nesting depth or platform start method.
`resolve_safe_execution_mode(requested_mode)` is the single chokepoint
`ParallelExecutionManager` consults before honoring a `"parallel"`
request: if already inside a worker process, the request is silently
downgraded to `"serial"` (logged, and recorded in the resulting
`ExecutionSummary.notes`) rather than rejected -- the safest available
behavior, since the inner work still completes correctly, just without a
nested pool.

`resolve_safe_project_execution(project)` applies the same protection to
a `Project`'s own Version 27 element-level `execution.mode="parallel"`
request, downgrading it to serial when the simulation task itself is
already running inside an orchestration worker.

## Performance metrics

Reuses Version 27's `femtoolkit.performance.benchmark.speedup` and
`parallel_efficiency` directly -- no new formula. For a batch that
actually ran in parallel:

$$
S = \frac{T_{serial,\,estimated}}{T_{parallel}} \qquad E = \frac{S}{P}
$$

where `T_serial,estimated = average_task_seconds * total_tasks` is an
*estimate* of the equivalent serial time (assuming uniform task cost,
ignoring process-startup/serialization overhead a real serial run would
not pay), not a second real measurement. For a batch that actually ran
serially, no such comparison applies, and `ExecutionSummary.parallel_speedup`/
`parallel_efficiency` are `None` -- never a fabricated `1.0`.

Amdahl's Law describes the theoretical ceiling for a workload with
parallelizable fraction `p`:

$$
S(P) = \frac{1}{(1-p) + p/P}
$$

**This is a ceiling, not a prediction.** Never expect a measured speedup
to match the theoretical value -- process-startup cost, serialization,
and per-task variance all pull the real number below it. See
`examples/orchestration/parallel_parameter_study.py` for a worked
comparison between an `ExecutionSummary`-estimated speedup and a
directly-measured wall-clock speedup, and why the two numbers usually
differ.

## Recommended worker counts and when parallel execution helps

- **One large batch of real FEA solves** (a parameter study with many
  scenarios, a Monte Carlo study with many samples): parallel execution
  reliably wins once each task's solve time exceeds the fixed cost of
  dispatching it to a worker -- a few hundredths of a second, in
  practice, for this toolkit's typical mesh sizes. `max_workers=4`
  (the default cap's usual value) is a reasonable starting point;
  `examples/orchestration/parallel_parameter_study.py` and
  `parallel_monte_carlo.py` both demonstrate a genuine, measured
  speedup at this scale.
- **A population-based optimization with many small generations**: each
  generation's offspring batch pays worker-pool-startup overhead
  separately (see "A fresh pool per batch" above), so a *small*
  population spread over *many* generations can end up slower in
  parallel than serial -- `examples/orchestration/parallel_optimization.py`
  shows this honestly, not hidden. A larger population evaluated over
  fewer generations amortizes that fixed cost far better.
- **Never assume more workers is always faster.** Diminishing returns
  (and even reversal) are expected once worker count exceeds the number
  of genuinely independent, CPU-bound tasks available, or once
  per-worker overhead dominates small per-task work.

## Limitations

- No distributed computing, MPI, cluster scheduling, or cloud execution
  -- this version is local-process-only. `ExecutionBackend` is kept
  generic enough that a future backend could add these without touching
  any call site built against the interface, but none is implemented
  here.
- No GPU/CUDA, no shared-memory FEA solver, no parallel sparse matrix
  assembly or parallel element-level computation (that remains Version
  27's separate, already-existing scope).
- `OrchestrationConfig.chunk_size` is defined and validated but not
  currently used by `LocalProcessBackend`, which submits one task per
  future individually (for per-task status/fail-fast/timeout
  granularity) rather than using `executor.map(..., chunksize=...)`.
  Reserved for a future version that batches dispatch.
- Timeout is a "no progress" approximation across the whole batch, not a
  true independent per-task wall-clock budget (see above).
- `random_search`/`coordinate_search` accept `orchestration_config` for
  interface consistency but do not batch their evaluations in this
  version -- coordinate search's moves are genuinely sequential/adaptive,
  and random search's independent-candidate batching is out of this
  version's scope (only the population-based algorithms the spec names
  explicitly gain parallel evaluation).

## Troubleshooting

- **`TaskSerializationError`**: a task's function or one of its
  arguments/results could not be pickled. The most common cause is a
  caller-supplied `lambda` or closure passed as an objective/constraint
  `evaluate` function -- replace it with a plain module-level function,
  or a small frozen `@dataclass` with a `__call__` method (see
  `femtoolkit.optimization.objectives._ExtractorObjective` for the
  pattern this toolkit's own builders use).
- **Parallel run is slower than serial**: check how many tasks are in
  each batch relative to how many sub-batches the workload is split
  into (see "Recommended worker counts" above) -- a small number of
  tasks per pool rarely amortizes process-startup cost.
- **A nested parallel request silently ran serially**: check
  `ExecutionSummary.notes` -- a downgrade due to nested-parallelism
  protection is always recorded there, with a matching message logged
  via the standard `logging` module (`femtoolkit.orchestration.nesting`).
- **GUI parallel execution still blocks the page**: this is expected in
  this version -- Streamlit has no background-task model wired in here;
  see `femtoolkit.gui.components.render_execution_mode_controls`'s
  docstring for the exact limitation.
