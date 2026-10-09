# Digital Twin & Model Updating (Version 38)

Answers a different question than Version 37's model fusion: *can the
expensive simulation model itself be corrected so it agrees better with
physical reality?* This document covers the digital twin concept, the
model-updating mathematics, model parameters, measurement data, the
calibration search, validation, the cantilever example, and the Digital
Twin Streamlit page. See the main [README](../README.md#version-38) for a
shorter overview.

## The engineering concept

A digital twin, in the foundational sense this version implements, connects
four things:

```text
Physical/Measured Data -> Simulation Model -> Parameter Updating -> Model Validation
```

Measured (or, for demonstration, clearly labeled synthetic) data is compared
against the simulation's own prediction at the same conditions. The
disagreement is used to adjust selected model parameters -- material
properties, geometric parameters, boundary-condition values -- so the
simulation's prediction moves closer to what was actually measured. The
updated model is then validated by comparing its accuracy, honestly, against
the original baseline.

## The model-updating mathematics

For a design vector of calibratable parameters `theta`, the simulation's
prediction at measurement point `i` is `y_sim,i(theta)`, and the corresponding
measured value is `y_meas,i`. The **residual** is

```text
r_i(theta) = y_sim,i(theta) - y_meas,i
```

and model updating searches for the parameter values that minimize the
**mean squared residual**

```text
J(theta) = (1/N) * sum_{i=1..N} (y_sim,i(theta) - y_meas,i)^2
```

subject to `theta_min <= theta <= theta_max` for every updatable parameter.
`J(theta)` is minimized by the *unmodified* Version 32/33 optimization
machinery (`OptimizationProblem`, `OptimizationRunner`,
`StopReason`) -- no second optimizer, and no Bayesian or machine-learning
calibration method, is implemented in this version.

## `ModelParameter`: the calibratable quantities

`ModelParameter` (`femtoolkit.digital_twin.parameters`) is the
engineering-facing description of one parameter under consideration: a
`name`, a `current_value` (the baseline guess), `lower_bound`/`upper_bound`,
`units`, an `updatable` flag, and a dotted override `path` (the same
convention `Scenario` uses, e.g. `"material.youngs_modulus"`). A non-updatable
parameter is reported alongside the updatable ones but never changed.
`to_design_variable()` converts an updatable parameter directly into a
Version 32 `DesignVariable` -- the exact bounded search every other
optimization problem in this toolkit uses, never a bespoke one.

## `MeasurementData`: the observed reference data

`MeasurementData`/`MeasurementPoint` (`femtoolkit.digital_twin.measurements`)
stores measured observations of one named engineering quantity --
`quantity_name` must be one of `femtoolkit.studies.extractors.EXTRACTORS`, so
the same named extractor already used throughout this toolkit reads the
matching simulation prediction. Each `MeasurementPoint` carries a
`measured_value`, the `input_conditions` it was taken under (e.g. a specific
applied load, keyed by override path), an optional `uncertainty`, and
free-form `metadata`. This is a lightweight, in-memory dataset -- **not** a
sensor-streaming system or a measurement database.

## `ModelUpdateProblem`: no new simulation logic

`ModelUpdateProblem` (`femtoolkit.digital_twin.problem`) ties a base project,
its `ModelParameter`s, and its `MeasurementData` together, failing fast
(`ValidationError`) on an empty parameter list, duplicate parameter names, no
measurement points, or an unknown `quantity_name`. `problem.predict(parameter_values)`
runs one simulation per measurement point -- applying each updatable
parameter's override and that point's own `input_conditions` -- by reusing
the *unmodified* Version 30 `Scenario`/`apply_scenario` override mechanism and
Version 34 `evaluate_simulation_batch`. No FEA solver logic lives in this
module; a point whose simulation does not complete, or whose response cannot
be extracted, is reported as `NaN` rather than silently dropped.

## `run_model_update`: the calibration search

```text
Initial parameters -> Run simulation -> Calculate error -> Modify parameters
      -> Run simulation again -> Calculate new error -> Keep improved parameters -> Repeat
```

`run_model_update` (`femtoolkit.digital_twin.updating`) converts every
updatable parameter into a `DesignVariable`, builds a picklable calibration
objective evaluating `J(theta)`, and hands both to the unmodified Version
32/33 `OptimizationRunner` -- `coordinate_search` (the default, a transparent,
deterministic hill-climbing search matching the modify/re-run/keep-if-improved
workflow above), `random_search`, or any Version 33 population-based
algorithm. The objective function stashes every measurement point's
prediction into `DesignContext.metadata` as it runs, so the full prediction
vector for both the baseline and the best-found design is recovered without a
second round of simulation after the search finishes -- the same pattern
Version 33's `robust.py` established for avoiding unnecessary repeated FEA.

`ModelUpdateResult` reports:

```text
initial_parameters / updated_parameters       every parameter's value before/after
initial_predictions / updated_predictions     simulation prediction at every measurement point, before/after
residuals_before / residuals_after            prediction - measured, before/after
convergence_history                           running best-found J(theta), per evaluation
status                                         CONVERGED / MAX_EVALUATIONS / FAILED / INVALID
```

`status` is mapped from the underlying Version 32/33 `StopReason`;
`convergence_history` reuses Version 32's
`OptimizationHistory.best_so_far_series` directly -- for a convergence plot,
not a new metric. `INVALID` means no updatable parameter was provided (no
simulation ran at all); `FAILED` means the baseline simulation itself did not
complete.

## The correction is never assumed to help

`compare_model_accuracy` (`femtoolkit.digital_twin.validation`) computes the
standard Version 35 `MetricSet` (MAE, RMSE, R^2, relative error) for the
baseline and the updated model, both against the same measured data:

```text
ModelAccuracyComparison.improved = after.rmse < before.rmse
```

A model update that makes agreement with measurements *worse* is reported
honestly, not hidden -- `improved` is a plain comparison, never an assumption.

## The cantilever beam example

`examples/digital_twin/cantilever_model_updating.py` generates a *synthetic*
"measured" tip displacement (clearly labeled as such -- not real experimental
data) from a cantilever whose true Young's modulus is 180 GPa, then runs model
updating starting from a baseline guess of 200 GPa. The search recovers a
value close to 180 GPa and reports the before/after MAE, RMSE, max error, and
relative error, concluding with whether agreement with the measurement
actually improved.

## The Digital Twin GUI page

The **Digital Twin** page (`femtoolkit.gui.workflow_pages.digital_twin_page`)
is one more page in the full engineering GUI, built the same way every page
consolidated in Version 37.1 already is: `AppState`, `require_project`,
`render_execution_mode_controls` (defaulting to Serial), and
`femtoolkit.application.exceptions_display.describe_error`, calling only
`femtoolkit.digital_twin` APIs with no measurement-handling, calibration, or
accuracy-metric logic of its own. Like the Adaptive Optimization and
Multi-Fidelity pages, it operates on whatever project is currently loaded
rather than a separate hardcoded example. The workflow: define baseline model
parameters, enter a measurement manually or **Generate Synthetic
Measurement** (always labeled **SYNTHETIC**, never presented as real data),
**Run Baseline Simulation**, **Run Model Updating**, and review the
comparison section's charts -- measured vs. simulated, residual before/after,
parameter before/after, the calibration objective before/after, and
model-update convergence. No expensive FEA runs automatically on a Streamlit
rerun; every step is gated behind its own button.

## Advantages and limitations

**Advantages.** Model updating turns a qualitative "the simulation doesn't
quite match reality" observation into a quantitative, repeatable calibration
procedure, reusing the existing optimization search rather than a new
estimation framework; the honest before/after accuracy comparison keeps the
engineer from assuming a correction helped just because it ran.

**Limitations.** No Bayesian calibration, machine-learning-based parameter
identification, real-time/IoT/sensor-streaming integration, state estimation,
Kalman filtering, structural health monitoring, anomaly detection, or any
industrial digital-twin platform feature (no live data connection, no cloud
component, no continuous/online updating loop) -- see the [Version 39
preview](../README.md#roadmap) for what builds on this foundation next.
`run_model_update` calibrates against a fixed, already-collected measurement
dataset; it does not run continuously against a live sensor feed. There is no
automatic acceptance criterion for "the update is good enough":
`compare_model_accuracy` reports the numbers, but judging whether an updated
model is acceptable for a given engineering use is left to the engineer.
