# Engineering Optimization & Design Exploration (Version 32/33)

Answers the question Version 30/31 left to a human running studies by
hand: *given a design space and an existing simulation, which designs
are worth looking at?* This document covers design variables, the
design space, objectives, constraints, feasibility, the six search
algorithms (two single-point, four population-based), multi-objective
optimization and Pareto fronts, robust (uncertainty-aware) design, and
reproducibility. See [`docs/studies.md`](studies.md) for the Version 30
parameter-study framework this version builds on and
[`docs/uncertainty.md`](uncertainty.md) for the Version 31 statistical
tools Version 33's robust design reuses directly, plus the main
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

Six algorithms -- two single-point (Version 32) and four population-based
including NSGA-II (Version 33, see below) -- all implement one shared
interface (`OptimizationAlgorithm.optimize`) so any algorithm can search
any `OptimizationProblem`. No SciPy dependency was added -- every
algorithm is a native implementation, kept deliberately simple enough
to read end to end.

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

For a population-based algorithm (Version 33), every evaluation also
carries its `generation` index (`None` for random/coordinate search,
which have no generation concept) and a `metadata` dictionary (used by
robust objectives/constraints to record diagnostic information -- see
"Robust (uncertainty-aware) design" below). `generation_summaries(history,
objectives)` aggregates per-generation statistics (evaluation/feasible/
infeasible/failed counts, that generation's own best feasible value, and
non-dominated front size among that generation's evaluations). **A
later generation's own best value is not guaranteed to improve on an
earlier one's** -- e.g. differential evolution only replaces a
population member when a trial strictly improves on it, so a generation
with few successful replacements is not evidence of regression. Track
the running best-so-far value via `compute_convergence` instead when
that distinction matters.

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
MAX_GENERATIONS  -- (Version 33) a population-based algorithm reached its
                     configured generation/iteration budget first.
TARGET_REACHED   -- (Version 33) the best-feasible objective value reached a
                     user-supplied `target_objective`; this reflects the
                     configured goal being met, not evidence of optimality.
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

Correlation is never used as an optimization gradient. Version 32's
`robust_objective_mean` (an optional, lightweight objective-function
builder returning a Monte Carlo mean) remains available unchanged;
Version 33's `femtoolkit.optimization.robust` generalizes this into a
full set of robust objective/constraint statistics (mean, percentile,
exceedance probability, ...) with explicit cost-control configuration
-- see "Robust (uncertainty-aware) design" above. `OptimizationMode.UNCERTAINTY_AWARE`
still exists on `OptimizationProblem` purely as an informational label;
it does not itself change how any search algorithm behaves -- genuine
uncertainty-aware evaluation always comes from how an objective or
constraint was built, never from this label alone.

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

An Optimization page: define design variables (with bounds, type,
units); optionally enable **Robust Design** (define uncertain
parameters, sample count, seed, sampling method, objective/constraint
statistic, percentile, and the safety ceiling, with the estimated total
FEA count shown live before running); define objectives (direction,
quantity -- automatically evaluated as a robust statistic instead of a
deterministic value when Robust Design is enabled) and constraints
(relation, limit); choose and configure an algorithm, including the
Version 33 population-based algorithms and NSGA-II (population size,
max generations, and each algorithm's own parameters -- mutation
factor/crossover probability for differential evolution; crossover/
mutation probability, tournament size, elite count for the genetic
algorithm; inertia weight, cognitive/social coefficients, velocity
limit for particle swarm); evaluate the baseline; run the search; and
review results -- for a single-objective problem, the best feasible
candidate compared against the baseline (explicitly labeled "best
feasible according to the defined objective and constraints," never
"best overall"), its robust-evaluation diagnostics when applicable, and
convergence plots (including a per-generation plot for population-based
runs, explicitly captioned "not necessarily monotonically improving");
for a multi-objective problem, the non-dominated set with no design
singled out as a winner.

## Population-based optimization (Version 33)

**Engineering concept.** A single-point search (random search,
coordinate search) only ever looks at the neighborhood of one current
design. A **population-based** algorithm instead maintains a whole set
of candidate designs at once, evolving that set across successive
**generations**:

- **Population** -- the set of candidate designs considered together
  at one generation.
- **Candidate solution** -- one design in the population, represented
  as a point in the design space.
- **Fitness/objective evaluation** -- scoring a candidate, exactly the
  same `evaluate_design` pipeline every algorithm in this package uses
  (no separate "fitness function" concept exists here).
- **Exploration** -- spreading the population across different regions
  of the design space, so the search does not get stuck near its
  starting point.
- **Exploitation** -- concentrating the population toward the better
  regions already found, refining a promising design further.
- **Mutation** -- a random perturbation that introduces new candidate
  designs the population would not otherwise reach.
- **Crossover** -- combining two existing candidates' design-variable
  values into a new candidate, mixing information already present in
  the population.
- **Selection** -- choosing which candidates survive, reproduce, or
  seed the next generation, almost always via the same
  feasibility-first comparison described above
  (`is_better_evaluation`), never a separately computed fitness score.

**Why this matters for engineering design problems.** A real FEA
objective (displacement, stress, mass as a function of geometry and
material) is rarely differentiable in a form usable by gradient-based
optimization, and a gradient may not exist at all for a discrete or
categorical design variable. A population-based algorithm needs no
gradient -- only the ability to evaluate a candidate design and compare
it to another, which `evaluate_design`/`is_better_evaluation` already
provide for every algorithm in this package, single-point or
population-based alike.

Mixed-type design variables (continuous, integer, categorical) are
searched by every population-based algorithm using one shared
real-valued "box" encoding (`femtoolkit.optimization.algorithms._encoding`):
a continuous or integer variable maps to its own `[lower, upper]`
range, and a categorical variable maps to `[0, len(categories) - 1]`,
the index into its category list -- the same "category as a list
position" idea coordinate search already uses, generalized to
continuous arithmetic.

### Differential evolution

```text
v = x_r1 + F * (x_r2 - x_r3)
```

For each target vector `x_i` in the population, three *other* distinct
population members `x_r1`, `x_r2`, `x_r3` are chosen at random. The
mutant vector `v` above uses the population's own spread as a
self-adapting step size: early on, when candidates are spread out,
steps are large (exploration); as the population converges, the same
formula automatically takes smaller steps (exploitation) -- no
step-size schedule needs to be hand-tuned. A trial vector is then built
by binomial crossover between `v` and `x_i` (each component taken from
`v` with probability `CR`, with at least one component always taken
from `v`), evaluated, and kept in place of `x_i` only if it is better
(feasibility-first).

Configurable: `population_size` (at least 4, since mutation needs
three *other* distinct members), `mutation_factor` (`F`, in `(0, 2]`),
`crossover_probability` (`CR`, in `[0, 1]`), `max_generations`,
`max_evaluations`, `seed`, and each design variable's own bounds.

### Genetic algorithm

```text
Elitism: copy the best `elite_count` individuals unchanged
Fill the rest of the next generation:
    Select two parents (tournament selection)
    Crossover (uniform, with probability crossover_probability)
    Mutation (per-gene random reset, with probability mutation_probability)
```

One selection strategy and one crossover/mutation pair, deliberately --
a clean, well-tested foundation rather than a large configurable
library of interchangeable operators. **Tournament selection** draws
`tournament_size` individuals at random and keeps the best one
(feasibility-first); **uniform crossover** builds a child by taking
each gene independently from one parent or the other; **random-reset
mutation** redraws a mutated gene uniformly within its own bounds
(coarser, and therefore typically less locally precise, than
differential evolution's differential-vector mutation or particle swarm's
velocity-based refinement -- a real, observed difference, not a defect).
**Elitism** copies the `elite_count` best individuals into the next
generation unchanged, so the best design found so far is never lost to
an unlucky generation of crossover/mutation.

Configurable: `population_size`, `crossover_probability`,
`mutation_probability`, `elite_count` (`0 <= elite_count <
population_size`), `tournament_size` (`2 <= tournament_size <=
population_size`), `max_generations`, `max_evaluations`, `seed`.

### Particle swarm optimization

```text
v_i <- w * v_i + c1 * r1 * (pbest_i - x_i) + c2 * r2 * (gbest - x_i)
x_i <- x_i + v_i
```

Each particle has a position (a candidate design) and a velocity. The
velocity update's three terms: `w * v_i` (**inertia**) keeps a particle
moving roughly the way it already was -- without it, particles would
jitter toward the best point found so far and lose any ability to
explore; `c1 * r1 * (pbest_i - x_i)` (the **cognitive** term) pulls a
particle back toward the best position *it personally* has found;
`c2 * r2 * (g - x_i)` (the **social** term) pulls every particle
toward the best position *the whole swarm* has found. `r1`/`r2` are
independent random numbers in `[0, 1)` redrawn every step. Velocity is
clamped to `velocity_limit` (a fraction of each variable's range, the
same convention `step_size` uses for coordinate search) before the
position update, and the resulting position is clipped back into
bounds.

Configurable: `population_size` (particle count), `inertia_weight`
(`w`, in `[0, 2]`), `cognitive_coefficient` (`c1`), `social_coefficient`
(`c2`), `velocity_limit` (in `(0, 1]`), `max_generations`,
`max_evaluations`, `seed`.

## Multi-objective optimization with NSGA-II (Version 33)

Version 32 already established Pareto dominance and `pareto_front`
(non-dominated filtering over a complete evaluation history). NSGA-II
(Deb et al., 2002) is a *search* algorithm specifically built to evolve
a population *toward* the Pareto front rather than hoping random or
single-point sampling happens to land near it.

**Non-dominated sorting** (`fast_non_dominated_sort`) partitions a
population into successive fronts: the first front is every design
nothing else in the population dominates; the second is every design
dominated only by first-front designs; and so on. Dominance here is
**constrained dominance** (`constrained_dominates`): a feasible design
always dominates an infeasible one regardless of objective values;
between two infeasible designs, the smaller total constraint violation
wins; two failed/invalid designs are never comparable -- the same
feasibility-first principle every algorithm in this package already
applies to single-objective comparisons, generalized to multiple
objectives.

**Crowding distance** (`crowding_distance`) measures how isolated a
design is within its own front: for each objective, the front is
sorted by that objective's value, the two boundary designs (best and
worst) get infinite distance (always preferred, to preserve the
trade-off curve's extremes), and every interior design accumulates the
normalized gap between its two neighbors. When only part of a front
can survive into the next generation, crowding distance decides which
subset survives -- keeping the population spread across the whole
trade-off curve rather than clustering in one region.

**One generation of NSGA-II:**

```text
Build an offspring population the same size as the parent population:
    Select two parents by binary tournament (lower rank wins;
    ties broken by larger crowding distance -- "less crowded")
    Crossover (uniform) + mutation (per-gene random reset)
Combine parents + offspring; re-rank into fronts; keep the best
`population_size` individuals, filling the last admitted front
by crowding distance (never arbitrarily)
```

Configurable: the same `population_size`, `crossover_probability`,
`mutation_probability`, `tournament_size`, `max_generations`,
`max_evaluations`, `seed` as the genetic algorithm. **NSGA-II never
selects one "best" solution** -- its job is to produce a good
approximation of the Pareto front; `OptimizationResult.pareto_front()`
reports the full non-dominated set from the complete run history, and
choosing a single preferred trade-off remains an engineering judgment
this package leaves to the caller.

## Robust (uncertainty-aware) design (Version 33)

**Deterministic vs. robust optimization.** Every algorithm above solves
`min f(x)` using one fixed set of input parameters per design -- a
*deterministic* optimization. A real engineering input is often
uncertain (material scatter, load variability); **robust optimization**
instead accounts for that uncertainty while still searching over the
same design variables `x`:

```text
min E[f(x, xi)]                          (minimize the expected objective)
Q_0.95(u(x, xi)) <= u_max                 (control a high percentile of a response)
P(sigma(x, xi) > sigma_allow) <= p_max    (limit an estimated exceedance probability)
```

where `xi` is an uncertain parameter
(`femtoolkit.uncertainty.parameters.UncertainParameter`, Version 31).
This toolkit does not implement a new *algorithm* for this -- every
algorithm above already works unchanged with any objective/constraint
function, deterministic or not. `femtoolkit.optimization.robust`
instead provides the **robust objective/constraint building blocks**:
`robust_objective_statistic`/`robust_constraint_statistic` run a small
Version 31 Monte Carlo study (reusing `MonteCarloRunner` directly, no
second execution path) at each design point and reduce the sampled
output to one statistic:

- `mean`, `median`, `std`, `coefficient_of_variation`, `min`, `max`
- `percentile` (with a configurable `percentile`, e.g. 95)
- `exceedance_probability` (with a configurable `threshold`/`direction`,
  reusing `femtoolkit.uncertainty.reliability.exceedance_probability`
  directly)

Every one of these is an **empirical** estimate from a finite Monte
Carlo sample at one design point -- never a rigorous reliability index
(no FORM/SORM is implemented anywhere in this toolkit). The statistic
used, its value, and the underlying sample count are recorded into
`DesignContext.metadata` and copied onto the resulting
`DesignEvaluation.metadata`, so a report or GUI can always show exactly
how a robust objective/constraint value was computed.

`RobustDesignConfig` (`uncertainty_enabled`, `sampling_method`,
`sample_count`, `random_seed`, `objective_statistic`,
`constraint_statistic`, `percentile`, `maximum_total_evaluations`,
`failure_policy`) is strictly opt-in -- `uncertainty_enabled=False` (the
default) means every objective/constraint is evaluated deterministically,
exactly as in Version 32. `OptimizationRunner.run()` accepts an optional
`robust_config` purely for cost-control validation and
reproducibility/reporting; it is carried onto `OptimizationResult` but
does not itself change how an objective or constraint is evaluated --
that is determined entirely by whether the objective/constraint was
built with `robust_objective_statistic`/`robust_constraint_statistic`.

### Nested simulation cost control

Robust optimization nests three loops:

```text
optimization evaluations
        |
        v
Monte Carlo samples
        |
        v
FEA simulations
```

A careless configuration (a large `max_evaluations` combined with a
large `sample_count`) can silently request an enormous number of FEA
solves. `estimate_total_fea_count(max_evaluations, robust_config)`
computes the upper bound (`max_evaluations * sample_count` when robust
evaluation is enabled), and `validate_robust_budget` -- called by
`OptimizationRunner.run()` *before* the baseline or any other
evaluation runs -- rejects a request exceeding
`robust_config.maximum_total_evaluations` with the same
`StudySizeExceededError` and "stop before execution, explain how to
reduce" pattern every prior safety check in this toolkit uses. A
requested budget is never silently reduced.

## Mathematical optimization benchmarks (Version 33)

`femtoolkit.optimization.benchmarks` provides three well-known
benchmark functions, each a plain NumPy function with **zero finite
element dependency** -- directly unit-testable on their own, used to
verify an algorithm's mechanics (does it converge toward a known
answer, does it respect bounds, is it reproducible) without the cost or
noise of a real FEA solve standing in for the objective's mathematical
behavior:

- **Sphere** -- `f(x) = sum(x_i^2)`, a smooth convex bowl, minimum
  `f(0, ..., 0) = 0`. The simplest possible sanity check.
- **Rosenbrock** -- `f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)`,
  a curved, narrow valley, minimum `f(1, ..., 1) = 0`. Easy to find the
  valley, much harder to converge precisely along it.
- **Rastrigin** -- `f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))`, a bowl
  covered in a regular lattice of local minima on top of the sphere
  shape, minimum `f(0, ..., 0) = 0`. Demonstrates a purely local search
  (coordinate search) getting trapped in a local minimum a
  population-based algorithm can escape.

`femtoolkit.optimization.benchmarks.problem_builder` is the (separate,
clearly-labeled) glue that wraps a benchmark function into a real
`OptimizationProblem`, so it can be run through the exact same
`OptimizationRunner` and every algorithm above -- each design variable
maps to one entry of a minimal project's `loads` list
(`loads.<i>.magnitude`, which accepts any finite real number, unlike a
positivity-constrained field such as thickness or Young's modulus).
Even a "pure math" benchmark still triggers one minimal, inexpensive
FEA solve per evaluation, since this toolkit has exactly one evaluation
pipeline -- the benchmark function itself never touches the FEA result.

See `examples/optimization/algorithm_benchmark_comparison.py` for all
six algorithms run against the same benchmark under the same budget and
seed, reported side by side with **no overall ranking** -- which
algorithm performs best is problem-dependent.

## Explicit scope exclusions (this version)

No machine learning, no topology optimization, no neural networks or
surrogate models, no commercial-solver integration, no Bayesian
optimization, no gradient-based or adjoint optimization, no automatic
CAD-driven shape optimization, no distributed/GPU/MPI/cluster
optimization, no FORM/SORM or other rigorous reliability method, and no
stochastic finite elements. See the Version 34 preview in the main
README for the planned next direction.

## Limitations

- **No distributed or parallel optimization execution.** Every
  evaluation runs sequentially; nothing here precludes wiring in
  parallel execution later, but none is implemented in this version.
- **No algorithm in this package makes a mathematical global-optimality
  guarantee.** Neither "converged," "completed," nor "max_generations"
  means a global optimum was found.
- **Coordinate search is a local search** and can stop at a local
  optimum (directly demonstrated in
  `examples/optimization/algorithm_benchmark_comparison.py`'s Rosenbrock
  comparison); its step-size-vs-domain-width sensitivity is a real,
  demonstrated limitation (see "Coordinate search" above).
- **The genetic algorithm's random-reset mutation is coarser than
  differential evolution's or particle swarm's refinement mechanisms**
  -- a real, observed precision difference on the sphere benchmark, not
  a defect; see "Genetic algorithm" above.
- **No multi-property categorical design variables** (e.g. full
  material catalog selection) -- see "Categorical design variables"
  above.
- **Every robust-design statistic is an empirical Monte Carlo estimate**
  from a finite sample at one design point -- never a rigorous
  reliability index. A small `sample_count` carries real sampling
  uncertainty, especially for `exceedance_probability` on a rare event.
- **`OptimizationMode.UNCERTAINTY_AWARE` remains an informational label**
  on `OptimizationProblem` -- it does not itself change search
  behavior; genuine uncertainty-aware evaluation comes entirely from
  building objectives/constraints with
  `robust_objective_statistic`/`robust_constraint_statistic`.
