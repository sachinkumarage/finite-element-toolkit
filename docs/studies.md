# Simulation Studies & Reproducible Workflows (Version 30)

Moves the toolkit from "one simulation -> one result -> one report" to
"simulation project -> scenarios -> multiple runs -> result comparison
-> verification/validation -> engineering report." This document covers
simulation projects, scenarios, parameter studies, run tracking and
history, result comparison, sensitivity analysis, reproducibility, and
study reporting. See [`docs/verification.md`](verification.md),
[`docs/validation.md`](validation.md), and
[`docs/reporting.md`](reporting.md) for the Version 29 framework this
version builds on, and the main [README](../README.md#version-30) for a
shorter overview.

## Engineering concepts

- **Simulation Model / Simulation Project** -- everything that defines
  one analysis: material, mesh, boundary conditions, loads, solver
  settings. This is exactly
  [`femtoolkit.application.project.Project`](../src/femtoolkit/application/project.py),
  introduced in Version 24. Version 30 does not duplicate it with a new
  "SimulationProject" class -- `Project` already is a structured,
  type-safe, JSON-serializable, versioned configuration; see "Why reuse
  `Project`?" below.
- **Simulation Scenario** -- one named "what if" variation of the base
  project: *what it changes*, not a full copy of every setting. A
  scenario's overrides are applied on top of an unmodified copy of the
  base project, so every scenario stays comparable back to the base
  case and to each other.
- **Parameter Study** -- "how does the result change as this parameter
  (or these parameters) varies?" Answered by auto-generating one
  scenario per combination of swept parameter values.
- **Simulation Run** -- one tracked execution of a project (with or
  without a scenario override): a run ID, status, timing, an immutable
  configuration snapshot, and (once finished) a result reference.

## Why reuse `Project` instead of a new "SimulationProject" model?

`Project` already satisfies nearly everything a "simulation project"
concept needs:

- Structured, type-safe sub-configs (`MaterialConfig`, `MeshConfig`,
  `BoundaryConditionConfig`, `LoadConfig`, `SolverConfig`,
  `ExecutionSettingsConfig`).
- `to_dict()`/`from_dict()` JSON serialization.
- A `format_version` field, with `Project.from_dict` already rejecting a
  file saved by a *future* toolkit version it does not know how to read
  (a minimal forward-compatibility foundation -- no migration framework
  is built yet, by design; see Limitations).
- Validation via `femtoolkit.application.validation.validate_project`
  (Young's modulus > 0, -1 < Poisson's ratio < 0.5, positive mesh size,
  positive solver tolerance, every value finite).

Version 30 adds exactly one new field to `Project`: `project_id`, a
UUID generated once at creation and preserved through save/load, giving
every project a stable identity a `SimulationRun` can reference. Every
new Version 30 capability is built **on top of** `Project` in two new
packages, `femtoolkit.runs` and `femtoolkit.studies`, rather than
recreating a parallel configuration system -- directly avoiding a
duplicated domain model.

## Architecture

```text
Project                                (femtoolkit.application.project -- Version 24)
   |
   |  Scenario.parameter_overrides / solver_overrides
   v
apply_scenario(base, scenario) -> Project     (femtoolkit.studies.scenarios)
   |
   v
SimulationRunManager.execute(project) -> SimulationRun   (femtoolkit.runs.manager)
   |    (thin wrapper around SimulationService.run -- Version 24, unchanged)
   v
StudyRunner.run(SimulationStudy) -> StudyResult   (femtoolkit.studies.runner)
   |
   +--> StudyResult.compare(...)      -> ComparisonResult   (femtoolkit.studies.comparison)
   +--> StudyResult.sensitivity(...)  -> [SensitivityResult] (femtoolkit.studies.sensitivity)
   +--> plot_study_quantity(...)      -> matplotlib Figure   (femtoolkit.studies.plots)
   +--> build_study_report(...)       -> StudyReport         (femtoolkit.studies.report)
```

`femtoolkit.runs.manager.SimulationRunManager.execute()` calls
`SimulationService.run()` directly -- the exact Version 24
validate -> build -> solve -> wrap pipeline every other part of this
toolkit uses -- and wraps the outcome into a `SimulationRun`. **No FEA
algorithm is duplicated anywhere in `femtoolkit.runs` or
`femtoolkit.studies`.**

## Scenarios

```python
from femtoolkit.studies import Scenario, apply_scenario

scenario = Scenario(
    scenario_id="aluminum",
    name="Aluminum variant",
    description="Same geometry and loading, aluminum material.",
    parameter_overrides={"material.youngs_modulus": 69e9, "material.poisson_ratio": 0.33},
    solver_overrides={"tolerance": 1e-8},
)
varied_project = apply_scenario(base_project, scenario)
```

`parameter_overrides`/`solver_overrides` keys are **dotted attribute
paths**, resolved against the project (`solver_overrides` paths are
relative to `project.solver`). A path may index into a list, e.g.
`"loads.0.magnitude"`. `apply_scenario` deep-copies `base_project`
first -- `base_project` is never mutated, and unrelated scenarios never
see each other's overrides. An invalid path (a field or index that does
not exist) raises `ValidationError` immediately, not a confusing
`AttributeError` deep inside a solve.

Every scenario in a study must have a unique `scenario_id`
(`validate_unique_scenario_ids`, raising
`DuplicateScenarioIdError` on a collision) so a run can always be traced
back to the scenario that produced it.

## Parameter sweeps

```python
from femtoolkit.studies import ParameterDefinition, generate_scenarios

load = ParameterDefinition(path="loads.0.magnitude", label="Tip load (N)", values=[-1000, -2000, -3000])
scenarios = generate_scenarios("Load Study", [load])   # 3 scenarios
```

Passing more than one `ParameterDefinition` produces the **Cartesian
product** of every parameter's values -- the same code path handles a
single-parameter sweep and a multi-parameter study. This grows fast: 3
parameters with 5 values each is 125 scenarios, not 15.

**`max_scenarios` stops a study before it runs, it never truncates one.**
`generate_scenarios(..., max_scenarios=100)` (the default) raises
`StudySizeExceededError` -- with a message explaining the actual count
and how to reduce it -- the moment the Cartesian product would exceed
the limit, before a single scenario is built or solved.

## Running a study

```python
from femtoolkit.studies import SimulationStudy, StudyRunner

study = SimulationStudy(
    study_id="load-study",
    name="Cantilever Load Study",
    base_project=base_project,
    parameters=[load],       # generates scenarios via the Cartesian product
    scenarios=[],             # explicit scenarios can be added alongside/instead
    max_scenarios=100,
)
result = StudyRunner().run(study)   # a StudyResult
```

`StudyRunner.run()` performs generate -> validate (unique IDs, scenario
count) -> execute -> collect. Execution is **strictly sequential** --
each scenario's project is built and solved one at a time via
`SimulationRunManager.execute()`. No parallel or distributed execution
is implemented in this version (see Limitations), though the loop's
structure -- one independent per-scenario step -- does not preclude
wiring in the Version 27 parallel execution infrastructure later.

A scenario that fails (an invalid override, e.g. a negative Young's
modulus) does not abort the study -- its run is recorded with
`status=FAILED`, `error_stage` (`"validation"` or `"solve"`), and
`error_message`, and the remaining scenarios still run.

`StudyResult` (serializable data, no behavior beyond thin query/compose
methods):

- `runs`, `successful_runs`, `failed_runs`
- `verification_summary()` / `validation_summary()` -- tallies of each
  run's real, already-computed status (never fabricated; see
  "Verification and validation integration" below)
- `compare(extractor, quantity_label)` -- see Result Comparison
- `sensitivity(parameter, extractor, quantity_label)` -- see Sensitivity

## Immutable configuration snapshots (reproducibility)

**A run must preserve its historical configuration even if the current
project configuration changes later.** `SimulationRunManager.execute()`
deep-copies the project into
`SimulationRun.configuration_snapshot` at the moment execution starts,
via `femtoolkit.runs.models.snapshot_configuration`. Every later change
to the live project (or to a shared base project across scenarios) has
no effect on a run that already holds its snapshot.

`Project` and its sub-configs are plain, mutable dataclasses (the GUI
assigns directly to fields like `project.material.youngs_modulus`), so
this "immutability" is a **policy** -- deep-copy once, then never
mutate -- enforced by `femtoolkit.runs`/`femtoolkit.studies`, not a
language-level (`frozen=True`) guarantee. Treat every
`configuration_snapshot` as read-only.

Each completed run also carries a
`ReproducibilityMetadata` record (`femtoolkit.reporting.metadata`,
Version 29) describing the toolkit/Python/dependency versions, model
configuration, and solver used -- collected automatically by
`SimulationRunManager`, no caller action needed.

## Run history

A lightweight, JSON-backed local persistence store -- **not** a server
database, per this version's explicit scope:

```python
from femtoolkit.runs import RunHistory, record_from_run

history = RunHistory()
for run in result.runs:
    history.add(record_from_run(run))   # a lightweight RunRecord, not the full result
history.save("run_history.json")

loaded = RunHistory.load("run_history.json")
loaded.by_project_id(project.project_id)
loaded.by_status("failed")
```

`RunRecord` intentionally holds only scalar key results, status,
timings, and the run's configuration snapshot as plain data -- never
the full field-by-field `SimulationRunResult` (numpy arrays, per-element
fields), keeping a history file small and reliably queryable. This
mirrors `ProjectService`'s exact `to_json`/`from_json`/`save`/`load`
persistence pattern (Version 24).

## Result comparison

**A different question from verification.**
`femtoolkit.verification.metrics` (Version 29) answers "how wrong is
this result compared to a reference?" with unsigned metrics -- a
numerical error's sign is not meaningful, only its magnitude.
`femtoolkit.studies.comparison` answers "how did the result change?"
with **signed** metrics -- the sign is the whole point (did
displacement go up or down when the load increased?):

```python
absolute_difference(v1, v2) == v2 - v1
relative_difference(v1, v2) == (v2 - v1) / max(|v1|, epsilon)
percentage_change(v1, v2) == relative_difference(v1, v2) * 100
```

```python
from femtoolkit.studies import get_extractor

displacement = get_extractor("maximum_displacement")
comparison = result.compare(displacement, "Maximum displacement")
for entry in comparison.entries:
    print(entry.run_id_2, entry.percentage_change)
```

`compare_runs`/`StudyResult.compare` compares every run against the
**first** run in the sequence as the baseline. `get_extractor` looks up
a named, reusable accessor
(`femtoolkit.studies.extractors.EXTRACTORS`: `maximum_displacement`,
`maximum_von_mises_stress`, `maximum_temperature`, `maximum_heat_flux`,
`solver_iterations`, `execution_time`) built directly on the existing
Version 22 `EngineeringSummary` fields -- no quantity is re-derived.

## Sensitivity analysis (a foundation, not a UQ framework)

```python
from femtoolkit.studies import compute_sensitivity, compute_sensitivity_series

result = compute_sensitivity(p1=1000, p2=2000, y1=0.5e-3, y2=1.0e-3)
result.sensitivity   # S = (dy/y) / (dp/p)
```

`S = (dy/y) / (dp/p)` is a first-order **finite-difference** ratio
between exactly two evaluated points -- a secant, not a derivative
computed in closed form. It describes local behavior between those two
points; it is not guaranteed representative elsewhere, especially
across a strongly nonlinear response.

A near-zero *reference value* (`p1` or `y1`) is floored by `epsilon`
(the same convention `femtoolkit.verification.metrics` uses). A
near-zero *actual parameter change* (`p2 == p1`) is different -- no
change was made, so there is nothing to compute a sensitivity from --
and raises `ValidationError` rather than being silently epsilon-floored
into a misleading number.

`StudyResult.sensitivity(parameter, extractor, quantity_label)` reads
each run's actual parameter value back from its immutable configuration
snapshot (not from `parameter.values`) and computes consecutive
pairwise sensitivities across the sweep
(`compute_sensitivity_series`) -- mirroring the point-to-point pattern
`MeshConvergenceStudy` (Version 29) already established.

**This is explicitly not uncertainty quantification.** No probability
distribution, confidence interval, or Monte Carlo sampling is involved
-- see the Version 31 preview in the main README.

## Verification and validation integration

- **Verification is never assumed.** `SimulationRun.verification_status`
  is taken directly from the run's own, already-automatically-computed
  Version 29 equilibrium check
  (`SimulationRunResult.equilibrium_check.status`) when the run
  completed, else `VerificationStatus.NOT_RUN`. A study result's
  `verification_summary()` is a tally of these real, per-run statuses
  -- never a blanket assumption that every scenario in a study is
  "verified."
- **Validation is never automatic.** `SimulationRun.validation_result`
  is `None` unless a caller explicitly supplies a
  `femtoolkit.validation.results.ValidationResult` compared against a
  real reference dataset (Version 29). `validation_summary()` reports
  `"not_validated"` for every run with no such comparison.

## Study reports

Extends the Version 29 reporting approach (`femtoolkit.reporting`) to a
study, in a thirteen-section document living in `femtoolkit.studies.report`
(kept out of `femtoolkit.reporting` to preserve a one-way dependency
direction: studies depends on reporting, never the reverse):

1. Study Summary
2. Base Model
3. Parameter Definitions
4. Scenarios
5. Run Status
6. Solver Information
7. Verification Summary
8. Validation Summary
9. Result Comparison
10. Sensitivity Results
11. Plots
12. Failed Runs
13. Reproducibility Metadata

```python
from femtoolkit.studies import build_study_report, render_study_report_markdown, save_study_report

report = build_study_report(
    title="Cantilever Load Study Report",
    study_summary="...",
    base_model_description="...",
    result=result,
    comparisons=[comparison],
    sensitivities=[sensitivity_series],
    conclusions="Displacement and stress scale linearly with load.",
)
save_study_report(report, "study_report.md", "markdown")
```

As with `EngineeringReport`, `conclusions` is free text the caller
supplies -- this framework never generates an engineering judgment
automatically.

## Recommended engineering workflow

```text
Create/Load Project (femtoolkit.application)
        |
        v
Define Parameter(s) or Explicit Scenario(s) (femtoolkit.studies.scenarios/parameter_sweep)
        |
        v
Review Scenario Count  <-- STOP if it exceeds max_scenarios; reduce before proceeding
        |
        v
Run Study (femtoolkit.studies.runner.StudyRunner -- sequential)
        |
        v
Inspect Run History & Failed Runs (femtoolkit.runs.history)
        |
        v
Compare Results (signed, femtoolkit.studies.comparison) + Sensitivity (femtoolkit.studies.sensitivity)
        |
        v
Verification Summary (real equilibrium checks) + Validation Summary (only if reference data supplied)
        |
        v
Generate Study Report (femtoolkit.studies.report) -- archive alongside the project file
```

## No material/design ranking

`femtoolkit.studies.comparison` reports absolute/relative/percentage
differences between runs -- it never synthesizes a "best" scenario or
material. See `examples/studies/material_comparison_study.py`: it
reports displacement, stress, density, and execution time side by side
for Steel/Aluminum/Titanium and explicitly does not declare a winner --
which material is appropriate depends on requirements (allowable
stress, weight, cost, environment) this toolkit has no way to know.

## Limitations

- **No distributed or parallel study execution.** Every scenario runs
  sequentially, in-process. The Version 27 parallel element-computation
  infrastructure is unrelated (it parallelizes within one solve, not
  across scenarios) and is not wired into study execution in this
  version.
- **No migration framework.** `Project.format_version` only rejects a
  file from a *future*, unrecognized format version; there is no
  automatic upgrade path from an older format yet -- "a clean
  foundation," not a complete migration system.
- **No SQL/server database.** `RunHistory` is a JSON file, chosen for
  simplicity and consistency with `ProjectService`'s existing
  persistence pattern; it is not built for concurrent multi-process
  writers or very large histories.
- **Sensitivity is a finite-difference foundation, not UQ.** No
  probability distributions, confidence intervals, or Monte Carlo
  sampling -- see the Version 31 preview.
- **No GPU/MPI/cluster execution**, by design, in this version.
