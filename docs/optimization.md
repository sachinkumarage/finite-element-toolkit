# Engineering Optimization & Design Exploration (Version 32)

Answers the question Version 30/31 left to a human running studies by
hand: *given a design space and an existing simulation, which designs
are worth looking at?* This document covers design variables, the
design space, objectives, constraints, feasibility, the two search
algorithms, multi-objective optimization and Pareto fronts, and
reproducibility. See [`docs/studies.md`](studies.md) for the Version 30
parameter-study framework this version builds on and
[`docs/uncertainty.md`](uncertainty.md) for the Version 31 statistical
tools it can optionally reference, plus the main
[README](../README.md#version-32) for a shorter overview.

## Design variables and the design space

A **design variable** is a parameter the optimization is allowed to
change -- a plate thickness, a beam diameter, a choice of element
formulation. `femtoolkit.optimization.variables.DesignVariable` pairs a
dotted override path (the same convention `Scenario`/
`ParameterDefinition`/`UncertainParameter` already use to locate a
project field) with a domain:

- **Continuous** -- `x in [lower, upper]`, e.g. a plate thickness in
  meters.
- **Integer** -- `n in {lower, ..., upper}`, e.g. a count of
  stiffeners.
- **Categorical** -- one of a fixed list of values, e.g. an element
  formulation choice. See "Categorical design variables" below for the
  current support boundary.

For variables `x1 in [a1, b1]`, `x2 in [a2, b2]`, ..., the **design
space** is the Cartesian product of every variable's domain -- every
combination an algorithm is allowed to propose. Optimization never
touches solver internals directly; it only ever proposes a point in
this space and asks the existing simulation pipeline to evaluate it.

```python
from femtoolkit.optimization import DesignVariable, DesignVariableType

thickness = DesignVariable(
    name="thickness", path="mesh.thickness",
    variable_type=DesignVariableType.CONTINUOUS,
    lower_bound=0.005, upper_bound=0.020, units="m",
)
```

## One simulation execution system, reused

```text
Design Variables -> Build Scenario -> Run FEA -> Extract Results ->
    Calculate Objectives -> Evaluate Constraints -> Design Evaluation
```

Every design evaluation becomes a `Scenario` (Version 30) with
`parameter_overrides` set from the design-variable values, applied via
`apply_scenario`, and executed with the exact same
`SimulationRunManager` every other part of the toolkit uses.
`femtoolkit.optimization` performs no FEA computation of its own and
implements no second execution engine.

## Objectives

```python
from femtoolkit.optimization import Objective, ObjectiveDirection, from_result_extractor
from femtoolkit.studies.extractors import get_extractor

displacement = Objective(
    name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
    evaluate=from_result_extractor(get_extractor("maximum_displacement")), units="m",
)
```

An `Objective` is never hard-coded into the optimization engine -- it
is a name, a `MINIMIZE`/`MAXIMIZE` direction (never hidden or
defaulted silently), units, and an evaluation function of a
`DesignContext` (the design-variable values, the resolved `Project`,
and the completed `SimulationRun`). `from_result_extractor` wraps any
Version 30 `Extractor`; `rectangular_mass` computes `width * height *
thickness * density` directly from the project configuration for
quantities that are not FEA outputs at all.

## Constraints

```python
from femtoolkit.optimization import Constraint, ConstraintRelation

stress_limit = Constraint(
    name="stress_limit", evaluate=from_result_extractor(get_extractor("maximum_von_mises_stress")),
    relation=ConstraintRelation.LESS_EQUAL, limit=250e6, units="Pa",
)
```

Three relations: `<=`, `>=`, `==` (each with a configurable tolerance).
Every constraint's **violation** is tracked independently:

```text
v = max(0, g(x))          # for a <= relation, g(x) = value - limit
```

Violations are never collapsed into one opaque penalty score -- a
report can always show exactly which constraint(s) a design failed and
by how much.

## Feasibility and design status

```text
FEASIBLE     -- every constraint satisfied; objective values are meaningful.
INFEASIBLE   -- the FEA run succeeded but at least one constraint is violated.
FAILED       -- the FEA run itself did not complete (solver failure, invalid
                geometry, numerical instability).
INVALID      -- the proposed design variable values are outside their
                declared domain; never sent to the solver at all.
```

A `FAILED` or `INVALID` design's objective values are always `None` --
never a fabricated or default number standing in for "no result."

> **A note on the specification's own wording.** Earlier drafts of this
> feature used "VALID" in one place and "FEASIBLE" in another for the
> same status. This toolkit standardizes on `DesignStatus.FEASIBLE`
> throughout the implementation.

## The baseline is never a candidate

Every optimization run evaluates the problem's **baseline** design (the
design variables' default values) once, separately, before the search
begins. The baseline is reported side by side with the best candidate
found, and every reported difference is a **signed, quantitative
comparison** (`OptimizationResult.improvement_over_baseline`, reusing
Version 30's `absolute_difference`/`percentage_change`) -- this toolkit
never automatically claims an optimized design is "better," only
reports by how much a quantity changed.

## Algorithms

Both algorithms implement one shared interface
(`OptimizationAlgorithm.optimize`) so any algorithm can search any
`OptimizationProblem`. No SciPy dependency was added -- both are native
implementations, kept deliberately simple enough to read end to end.

### Bounded random search

Draws `max_evaluations` points uniformly at random from the design
space (`numpy.random.default_rng(config.seed)`, never global random
state), keeping the best feasibility-ranked design seen so far. The
simplest possible baseline algorithm, fully reproducible given a seed.

### Coordinate search

```text
Start at the baseline design.
For each design variable, try current value +/- step:
    Evaluate the +step candidate, evaluate the -step candidate.
    Keep whichever (including staying put) is best.
Repeat full passes over every variable until a pass makes no
improvement, or another stopping condition is reached.
```

A step moves a **continuous** variable by `step_size` (a *fraction* of
its `[lower, upper]` span, clamped at the bounds), an **integer**
variable by at least one integer step scaled the same way, and a
**categorical** variable to the next/previous entry in its category
list (no wraparound -- the ends of the list are the ends of the
search). Reuses the baseline evaluation as its starting point rather
than re-running it.

**A real, discovered limitation.** A `step_size` that is too coarse
relative to an integer variable's domain can strand the search away
from the true optimum: on a `[1, 20]` integer domain, `step_size=0.5`
produces a jump of `10`, which can overshoot past the optimum in one
step and never return to it, while `step_size=0.1` (a jump of `2`)
converges correctly. Choose `step_size` with the coarsest variable's
domain width in mind, not just the finest one.

Coordinate search is a **local, derivative-free** search: it can stop
at a local optimum a broader search might avoid, and it says nothing
about variable combinations an early-stopped pass never explored. No
mathematical global-convergence guarantee is made or implied anywhere
in this toolkit for either algorithm.

## Optimization history

Every single evaluation -- feasible, infeasible, failed, or invalid --
is recorded in `OptimizationHistory`: evaluation number, design
variable values, objective values, constraint evaluations,
feasibility, execution time, and (for FEA-backed evaluations) the
underlying run's status. `best_so_far_series(objective)` traces how the
best feasible value seen evolved evaluation by evaluation. This history
is always fully inspectable -- never a black box summarized only as a
final answer.

## Convergence and stopping criteria

```text
delta_f     = |f_best,k - f_best,k-1|
delta_f_rel = delta_f / |f_best,k-1|   (when f_best,k-1 != 0)
```

`compute_convergence` checks whether the last `patience` evaluations'
relative improvement all stayed below `config.tolerance`. A run stops
for exactly one recorded reason:

```text
COMPLETED        -- coordinate search finished a full pass with no improvement.
MAX_EVALUATIONS  -- the evaluation budget was reached.
CONVERGED        -- objective improvement stayed below tolerance for `patience`
                     consecutive evaluations.
CANCELLED        -- the run was cancelled (GUI execution).
FAILED           -- the consecutive-failure threshold was exceeded.
```

`OptimizationResult.stop_reason` and every rendered report always state
this plainly -- "converged" here means *this heuristic search's own
improvement criterion was satisfied*, never "the global optimum was
found."

## Constraint handling: feasibility-first, never a hidden penalty score

`is_better_evaluation(candidate, incumbent, objective)` implements one
transparent, three-step comparison, used identically by both
algorithms:

1. **Feasibility rank first.** `FEASIBLE < INFEASIBLE < FAILED/INVALID`
   (the latter two tied). A feasible design always beats an infeasible
   one regardless of objective value.
2. **Among equally-feasible designs, compare the objective** per its
   `MINIMIZE`/`MAXIMIZE` direction.
3. **Among equally-infeasible designs, compare `total_violation`**
   (the sum of every constraint's violation) only as a secondary
   tie-breaker -- never combined with the objective into one score.
4. Two `FAILED`/`INVALID` designs are never considered comparable; the
   current incumbent is always kept.

No configurable penalty-score constraint handling is implemented in
this version -- feasibility-first is the only strategy, and it is
never hidden behind a black-box combined fitness value.

## Multi-objective optimization: objective vectors and Pareto dominance

An `OptimizationProblem` with more than one `Objective` is
multi-objective (`problem.is_multi_objective`). For a stopping decision
only, both algorithms track progress against `problem.objectives[0]` as
a documented practical proxy -- the actual multi-objective **result**
is always the complete, independently computed Pareto front over the
full history, never this internal proxy.

**Pareto dominance.** Design `A` dominates design `B` if `A` is no
worse than `B` on every objective and strictly better on at least one
(respecting each objective's own `MINIMIZE`/`MAXIMIZE` direction). Two
designs with identical objective vectors do not dominate each other --
both remain in the non-dominated set.

```python
from femtoolkit.optimization import pareto_front

front = pareto_front(result.history.evaluations, [mass_objective, displacement_objective])
```

`pareto_front` filters to `FEASIBLE` evaluations only (infeasible,
failed, and invalid designs are never part of a Pareto front) and never
sorts or ranks the resulting set -- **no design in the non-dominated
set is ever labeled "best overall."** Choosing among Pareto-optimal
trade-offs is an engineering judgment this toolkit deliberately leaves
to the engineer.

A single-objective "Pareto front" is a well-defined, non-trivial
concept too -- exactly the evaluation(s) tied for the best objective
value, not an empty set.

## Design-space exploration

`OptimizationHistory` doubles as a design-space exploration record:
every sampled design's variable values, objective values, constraint
values, and feasibility are available for inspection regardless of
which algorithm produced them, integrating naturally with the Version
30 parameter-study infrastructure this package builds on. No surrogate
model of any kind is fit to this data in this version.

## Relationship to Version 31 uncertainty analysis

Optimization and uncertainty analysis answer different questions and
are kept clearly separate:

- **Statistical sensitivity** (Version 30/31, `dy/y / dp/p` or Monte
  Carlo correlation) describes how an *existing* design's output
  responds to input variation.
- **Optimization direction** (this version) is a search decision made
  by an algorithm proposing new candidate designs.

Correlation is never used as an optimization gradient. The one
integration point is `robust_objective_mean`, an optional, explicitly
lightweight objective-function builder that runs a real Version 31
`MonteCarloConfig`/`MonteCarloRunner` study at a given design point and
returns the mean of a named output quantity -- useful as one building
block toward uncertainty-aware design, but not itself a robust
optimization algorithm. `OptimizationMode.UNCERTAINTY_AWARE` exists on
`OptimizationProblem` purely as an informational label in this version;
it does not change how either search algorithm behaves.

## Categorical design variables

`DesignVariableType.CATEGORICAL` works cleanly wherever a single
project field genuinely accepts a discrete choice through the existing
one-path-per-variable override mechanism -- see
`examples/optimization/element_type_selection.py`, which optimizes over
`mesh.element_type in {"quad", "cst"}`.

**Not yet supported: multi-property material catalog selection.** A
choice like "Steel vs. Aluminum vs. Titanium" must change *several*
project fields (Young's modulus, Poisson's ratio, density) atomically
for one categorical decision. `DesignVariable` is deliberately a
single-path abstraction, matching `Scenario`/`ParameterDefinition`, and
extending it to atomically drive a *group* of fields is real future
work rather than something forced into this version's type system.

## Evaluation budget and safety limits

`OptimizationConfig.__post_init__` rejects `max_evaluations` above a
configurable `evaluation_limit` (default 1000) immediately, reusing
Version 30's `StudySizeExceededError` and its exact "stop before
execution, explain how to reduce" pattern -- a requested budget is
never silently reduced. `OptimizationConfig.estimated_simulation_count`
gives an honest, simple estimate (`= max_evaluations`) for the
deterministic case; it does not account for the extra internal FEA runs
an objective like `robust_objective_mean` performs per evaluation.

## Failed evaluation handling

A design can fail for several distinct reasons -- invalid geometry or
material parameters, mesh generation failure, solver non-convergence,
or numerical instability. Every failure is recorded in the history with
its `error_message`, and a failed evaluation's objective values are
always `None`, never treated as a valid (e.g. zero, or worst-case)
number by any comparison or report.

## Optimization reports

`femtoolkit.optimization.report` renders a twenty-section document:

1. Study Summary
2. Base Model
3. Problem Definition
4. Design Variables
5. Objectives
6. Constraints
7. Algorithm & Configuration
8. Baseline Design
9. Optimization History Summary
10. Best Feasible Candidate(s)
11. Improvement Over Baseline
12. Constraint Satisfaction
13. Convergence
14. Stopping Reason
15. Pareto Front (multi-objective only)
16. Design-Space Coverage
17. Plots
18. Verification Status
19. Reproducibility Metadata
20. Limitations

Section 20 always states the stop reason's plain-language meaning and a
fixed disclaimer: both algorithms are derivative-free heuristics with
no convergence guarantee, coordinate search can stop at a local
optimum, every reported improvement is a quantitative baseline
comparison only, and no distributed, parallel, or GPU optimization was
used -- every evaluation ran sequentially.

```python
from femtoolkit.optimization import build_optimization_report, save_optimization_report

report = build_optimization_report(
    title="...", summary="...", base_model_description="...", result=result, conclusions="...",
)
save_optimization_report(report, "report.md", "markdown")
```

As with every prior reporting module in this toolkit, `conclusions` is
free text the caller supplies -- never generated automatically.

## Reproducibility

Every `OptimizationResult` carries what is needed to reproduce a run
"subject to deterministic numerical execution": the full
`OptimizationConfig` (algorithm, max evaluations, tolerance, seed, step
size), every `DesignVariable` definition (bounds, type, units,
default), every `Objective` and `Constraint` definition, and the
complete `OptimizationHistory` (every evaluation, feasible or not). No
duplicate FEA result-array storage is introduced -- results reuse the
Version 30 `SimulationRun`/`Project` types directly.

## GUI integration

A new Optimization page: define design variables (with bounds, type,
units), objectives (direction, quantity), and constraints (relation,
limit); choose and configure an algorithm (max evaluations, tolerance,
seed); evaluate the baseline; run the search with a progress display;
and review results -- for a single-objective problem, the best feasible
candidate compared against the baseline (explicitly labeled "best
feasible according to the defined objective and constraints," never
"best overall"); for a multi-objective problem, the non-dominated set
with no design singled out as a winner.

## Explicit scope exclusions (this version)

No machine learning, no topology optimization, no neural networks or
surrogate models, no commercial-solver integration, no genetic
algorithms, no particle swarm optimization, no gradient-based or
adjoint optimization, no automatic CAD-driven shape optimization, no
distributed/GPU/MPI/cluster optimization, and no full robust
optimization or FORM/SORM reliability-based optimization. See the
Version 33 preview in the main README for the planned next direction.

## Limitations

- **No distributed or parallel optimization execution.** Every
  evaluation runs sequentially; nothing here precludes wiring in
  parallel execution later, but none is implemented in this version.
- **Both algorithms are derivative-free heuristics with no
  mathematical global-optimality guarantee.** Neither "converged" nor
  "completed" means a global optimum was found.
- **Coordinate search is a local search** and can stop at a local
  optimum; its step-size-vs-domain-width sensitivity is a real,
  demonstrated limitation (see "Coordinate search" above).
- **Only two algorithms are implemented.** No differential evolution,
  genetic algorithms, particle swarm optimization, or gradient-based
  methods in this version.
- **No multi-property categorical design variables** (e.g. full
  material catalog selection) -- see "Categorical design variables"
  above.
- **`OptimizationMode.UNCERTAINTY_AWARE` is informational only** in
  this version; it does not change search behavior.
