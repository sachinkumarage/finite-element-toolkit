# Multi-Fidelity Modeling (Version 37)

Answers a different question than Version 36's adaptive sampling: *can a cheap,
approximate model and the expensive high-fidelity model be combined so the cheap
model's systematic error is corrected?* This document covers fidelity levels and
models, the paired dataset, model discrepancy, the discrepancy surrogate, the
fused prediction, high-fidelity verification, and the Multi-Fidelity Streamlit
page. See the main [README](../README.md#version-37) for a shorter overview.

## Low fidelity vs. high fidelity

```text
Low Fidelity   -> fast but less accurate
High Fidelity  -> slower but more accurate
```

The purpose is to use the cheap model for screening and the expensive model for
verification -- not to replace the expensive model outright.

## Model discrepancy

For a design vector `x`, let `y_L(x)` be the low-fidelity result and `y_H(x)`
the high-fidelity result. The **model discrepancy** is

```text
delta(x) = y_H(x) - y_L(x)
```

and the simplest fused model -- the only method this version implements -- adds
a surrogate of that discrepancy back onto the cheap result:

```text
y_hat_H(x) = y_L(x) + delta_hat(x)
```

where `delta_hat` is a surrogate model of `delta`, trained with the completely
unmodified Version 35 surrogate machinery.

## Fidelity levels and models

`FidelityLevel` (`femtoolkit.multifidelity.fidelity`) is a named, extensible
label with a relative `estimated_cost` -- `LOW_FIDELITY` and `HIGH_FIDELITY` are
provided; a caller can define another level without this toolkit needing a
complex fidelity hierarchy. `FidelityModel` is the abstraction that actually
evaluates a design point at one fidelity level:

- `AnalyticalFidelityModel` wraps a plain, cheap callable (e.g. an
  Euler-Bernoulli beam formula) -- no FEA solve at all.
- `SimulationFidelityModel` wraps the unmodified Version 30
  `Scenario`/`apply_scenario` override mechanism and `SimulationRunManager` --
  the real solver is reused exactly as every other version reuses it, never
  duplicated.

`summarize_costs(models)` returns a simple textual/tabular cost-and-accuracy
comparison (spec section 13) -- not a cost-optimization algorithm.

## The paired dataset

`MultiFidelityDataset` (`femtoolkit.multifidelity.dataset`) stores
`MultiFidelitySample` objects, each an `inputs` dict plus a `low_result` and an
optional `high_result`. A sample is **paired** once both results exist.
`compute_discrepancy` validates that the two results cover exactly the same
response names (and contain only finite values) before subtracting --
`IncompatibleFidelityResultError` otherwise. `dataset.discrepancies()` returns
every paired sample's discrepancy as a NumPy array, in order.

**Reuse, not a second dataset system.**
`dataset.to_discrepancy_dataset()` converts every paired sample into a
`femtoolkit.surrogate.datasets.SnapshotDataset` of `x -> delta(x)` -- the exact
same dataset class Version 35 surrogate training already expects.

## The discrepancy surrogate

`train_discrepancy_surrogate(dataset, config)`
(`femtoolkit.multifidelity.discrepancy`) is a thin wrapper: it calls
`dataset.to_discrepancy_dataset()` and then the unmodified Version 35
`train_surrogate`. Any supported surrogate type
(`femtoolkit.surrogate.models.SURROGATE_MODEL_TYPES` -- currently
`"polynomial"`/`"rbf"`) can be selected via `TrainingConfig.model_type`; no new
regression or machine-learning framework is introduced.

## The fused prediction

`MultiFidelityModel` (`femtoolkit.multifidelity.model`) combines a
`FidelityModel` (low fidelity) with a fitted discrepancy surrogate.
`model.predict(point)` returns a `FusedPrediction` with three separate fields:

```text
low_fidelity_result      y_L(x)
predicted_discrepancy    delta_hat(x)
fused_prediction         y_hat_H(x) = y_L(x) + delta_hat(x)
```

`is_fused_prediction = True` is always set, mirroring Version 35's
`SurrogatePrediction.is_surrogate_prediction` -- **a fused prediction is never
the same thing as an actual high-fidelity FEA result.**

## High-fidelity verification

`verify_fused_prediction(model, high_fidelity_model, points)` both predicts
every point (cheap: a low-fidelity evaluation plus a discrepancy-surrogate
prediction) and actually evaluates `high_fidelity_model` at the same point. Each
`FusionVerificationRecord` reports:

```text
e_abs = |y_H - y_hat_H|
e_rel = |y_H - y_hat_H| / (|y_H| + epsilon)
```

for both the raw low-fidelity result and the fused prediction, plus a
`FusionAcceptanceStatus`:

```text
IMPROVED               -- the fused prediction's error is <= the low-fidelity error
NOT_IMPROVED           -- the correction did not help (or made things worse)
VERIFICATION_FAILED    -- the high-fidelity run itself did not complete
```

**The correction is never assumed to help.** `compare_fidelity_accuracy(model, dataset)`
(`femtoolkit.multifidelity.validation`) computes the standard Version 35
`MetricSet` (MAE, RMSE, R², relative error) for the low-fidelity-only result and
for the fused prediction, both against the same high-fidelity samples --
`report.improves_on_low_fidelity(response_name)` is a simple RMSE comparison,
reported honestly rather than assumed.

## The cantilever beam example

`examples/multifidelity/cantilever_multifidelity_fusion.py` compares a cheap
Euler-Bernoulli beam formula (ignoring shear deformation) against the existing
continuum FEA cantilever model, trains a discrepancy surrogate, and reports
whether the fused prediction is actually more accurate than the raw analytical
formula on held-out thickness values -- it reliably is, since the beam formula's
error (neglected shear deformation) is a smooth, learnable function of
thickness.

## The Multi-Fidelity Streamlit page

Extends the existing Version 36 quickstart app (`streamlit_app.py`) with one
more page -- no new Streamlit architecture. The workflow: configure the
low-/high-fidelity models and design parameters, **Generate Samples**, review
the discrepancy, select a surrogate method and **Train Discrepancy Model**,
then **Predict** and optionally **Run High-Fidelity Verification** at a query
point. Every result is explicitly labeled `LOW-FIDELITY RESULT`,
`MULTI-FIDELITY PREDICTION`, `HIGH-FIDELITY FEA`, or `VERIFICATION RESULT` --
never blurred together. No expensive FEA runs automatically on a Streamlit
rerun; every step is gated behind its own button.

## Advantages and limitations

**Advantages.** Screening with the cheap model is fast; the discrepancy
correction can recover much of the high-fidelity accuracy without paying for a
high-fidelity solve at every point; the cost comparison and validation metrics
make the trade-off explicit rather than hidden.

**Limitations.** No Gaussian processes, co-kriging, deep learning, neural
networks, Bayesian optimization, topology/shape optimization, digital twins,
distributed/GPU/cloud computing, advanced reliability methods, complex
multi-fidelity optimization, or adaptive fidelity selection. Only the simple
additive correction is implemented. There is no automatic acceptance criterion
for "the fusion is good enough" -- `compare_fidelity_accuracy`/
`verify_fused_prediction` report the numbers; judging whether a fused model is
acceptable for a given engineering use is left to the caller. See the [Version
38 preview](../README.md#roadmap) for what builds on this foundation next.
