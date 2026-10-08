# Reduced-Order Modeling & Surrogate-Based Engineering Analysis (Version 35)

Answers the question every prior version left expensive: *once a real
FEA model exists, how can it be approximated cheaply enough for the
repeated evaluations a Monte Carlo study, an optimization search, or a
design-space sweep needs?* This document covers snapshot datasets,
feature/response scaling, the surrogate-model abstraction (polynomial
regression and radial basis functions), validation metrics and
cross-validation, Proper Orthogonal Decomposition (POD) reduced-order
modeling, the applicability-domain foundation, and how this layer
integrates with the Version 30 simulation workflow, Version 31
uncertainty quantification, Version 33 optimization, and Version 34
parallel execution -- without ever letting a surrogate silently replace
the high-fidelity model. See the main
[README](../README.md#version-35) for a shorter overview.

## The high-fidelity model

Every analysis so far in this toolkit computes

```text
y = f(x)
```

where `x` is a vector of input/design variables (Young's modulus,
density, thickness, load, temperature, ...) and `f` is the expensive
FEA simulation itself. `y` is one or more scalar engineering responses:
maximum displacement, maximum stress, a reaction force, a natural
frequency, a temperature, a heat flux, a mass. `f` is never replaced by
this package -- only approximated, with every approximation clearly
labeled as such.

## Reduced-order modeling vs. surrogate modeling

Two distinct approximation strategies are provided, for two distinct
kinds of quantity:

- **Reduced-order modeling** (`femtoolkit.surrogate.rom`) approximates
  a *full-field* solution (e.g. every nodal displacement DOF) with a
  much smaller representation, `u ~ V q`, where `V` is a reduced basis
  and `q` is a small vector of reduced coordinates.
- **Surrogate modeling** (`femtoolkit.surrogate.models`) approximates a
  *scalar* response directly as a function of the design variables,
  `y ~ g(x)`, without reducing a field at all.

```text
High-Fidelity FEA
        |
        v
  Snapshot Data
        |
        v
Reduced Basis / Surrogate Model
        |
        v
  Fast Prediction
```

## Snapshots and datasets

A `Snapshot` (`femtoolkit.surrogate.datasets`) is one high-fidelity
evaluation: an input/design point paired with the response(s) it
produced. A `SnapshotDataset` is an order-preserving collection of
snapshots that validates every addition against its declared
`feature_names`/`response_names`, so the input-output correspondence
can never silently become misaligned (`InconsistentSnapshotError` if a
snapshot is missing a declared name, carries an extra one, or contains
a non-finite value).

`collect_snapshots_from_runs(runs, design_points, response_quantities)`
builds a dataset directly from already-executed
`femtoolkit.runs.models.SimulationRun` objects -- the same objects the
Version 30 study/Version 34 orchestration infrastructure already
produces. `dataset.split(train_fraction, validation_fraction, seed=...)`
returns a reproducible `DatasetSplit` (train/validation/test), and
`dataset.with_additional_snapshots(new_snapshots)` returns a *new*
dataset with `dataset_version` incremented (`v1 -> v2`) rather than
mutating the original in place -- a previously trained model's recorded
dataset version therefore never silently loses its meaning.

## Feature and response scaling

`femtoolkit.surrogate.scaling` provides `StandardScaler`
(zero mean, unit variance) and `MinMaxScaler` (`[0, 1]`), behind one
`Scaler` interface (`fit`, `transform`, `inverse_transform`,
`fit_transform`). Every `SurrogateModel` owns its own feature scaler and
response scaler internally, fit on training data only -- a prediction
is always returned in physical units via `inverse_transform`, never as
a raw normalized value a caller could mistake for an engineering
result.

## Surrogate models

`SurrogateModel` (`femtoolkit.surrogate.models.base`) is the shared
abstraction: `fit`, `predict`, `predict_point`, `score`, `validate`,
`serialize`/`load_state`. Two concrete, transparent models are
provided (`SURROGATE_MODEL_TYPES = ("polynomial", "rbf")`):

- `PolynomialRegressionSurrogate(degree=1 | 2)` -- ordinary least
  squares on the polynomial feature expansion (linear or full
  quadratic, including cross terms), solved with `numpy.linalg.lstsq`
  rather than a direct (potentially singular) matrix inverse.
- `RBFSurrogate(kernel="gaussian" | "multiquadric", epsilon=None, regularization=1e-8)`
  -- a radial basis function interpolant, `f(x) = sum_i w_i * phi(||x - x_i||)`,
  with a data-driven default shape parameter and Tikhonov
  regularization for numerical stability.

`predict_point(point)` is the entry point application code should
prefer whenever a result may reach an engineering user: it always
returns a `SurrogatePrediction` carrying the predicted value(s),
model/dataset identity, scaling configuration, applicability-domain
status, and an explicit `is_surrogate_prediction = True` flag.

## Validation: never trust training-set accuracy

`train_surrogate(dataset, config)` (`femtoolkit.surrogate.workflows.training`)
splits, fits, and validates a surrogate in one call, returning
`(model, SurrogateValidationReport)`. The report never evaluates a
model only on its own training data -- it always reports held-out
validation/test metrics separately, plus optional `k_fold_cross_validate`
results (fold scores, mean, standard deviation) and
`EngineeringTolerance` checks (`max_absolute_error`/`max_relative_error`
per response).

**Statistical accuracy is not engineering acceptability.** A surrogate
can report an excellent global R^2 and still fail a safety-critical
response's worst-case error. `SurrogateValidationReport.meets_engineering_tolerances`
is `True` only if every *configured* tolerance passed -- it is
vacuously `True` with no tolerance configured, and that vacuous case is
never interpreted elsewhere as "the model is safe."

## Proper Orthogonal Decomposition (POD)

`PODModel` (`femtoolkit.surrogate.rom.pod`) computes the SVD of a
snapshot matrix `X = U Sigma V^T` (one full-field snapshot per column)
and retains the first `r` columns of `U` as the reduced basis.
Selection is either a **fixed rank** (`rank=10`) or the **minimum rank
reaching an energy threshold** (`energy_threshold=0.999`), never both.
`captured_energy`/`discarded_energy` report the fraction of the
snapshot set's Frobenius-norm "energy" the truncated basis reproduces
exactly on the training snapshots; `energy_spectrum()` returns the full
cumulative-energy curve for a convergence plot.

`femtoolkit.surrogate.rom.snapshots` collects `FieldSnapshot` objects
(a full-field vector plus the design point that produced it) from
already-executed simulation runs via a caller-supplied field extractor,
and `build_snapshot_matrix` stacks them into `X` (raising
`InconsistentSnapshotError` if the snapshots do not share the same
field dimension). Reconstruction (`pod.reduce`/`pod.reconstruct`) on
held-out snapshots the basis never saw is the only honest way to assess
generalization -- training-set reconstruction error is never assumed to
carry over. See `examples/surrogate/pod_cantilever_displacement_field.py`.

## Applicability domain

`ApplicabilityDomain` (`femtoolkit.surrogate.domain`), built from the
raw training inputs, checks a query point against each feature's
training bounds and reports one of four `DomainStatus` values:
`WITHIN_TRAINING_DOMAIN`, `BOUNDARY`, `OUTSIDE_TRAINING_DOMAIN`, or
`INVALID` (a missing/extra feature or a non-finite value). Every
`SurrogatePrediction` carries this status -- a surrogate is never
silently trusted to extrapolate confidently.

## Adaptive sampling (foundation only)

`recommend_candidates(model, candidate_points)`
(`femtoolkit.surrogate.workflows.recommendation`) ranks a caller-supplied
candidate pool by how poorly covered each point is by the training
data, returning `SamplingCandidate` objects worth a future high-fidelity
evaluation. It never generates candidate points itself and never
triggers a new simulation on its own -- a human or calling study must
explicitly approve and run any recommended point, then retrain:

```text
Surrogate -> Identify candidate -> Recommend FEA evaluation ->
User/study approves -> High-fidelity FEA -> Add snapshot -> Retrain
```

## Integration with Version 30/31/33/34

- **Version 30/34.** `generate_training_dataset(base_project, parameters, response_quantities, orchestration_config=...)`
  (`femtoolkit.surrogate.workflows.training`) builds one scenario per
  parameter-sweep combination (Version 30) and executes the whole batch
  through `evaluate_simulation_batch` (Version 34) -- serial by
  default, or parallel if an `OrchestrationConfig` is supplied. No
  second parallel-execution system is introduced.
- **Version 31.** `EvaluationBackend` (`HIGH_FIDELITY`/`SURROGATE`,
  `femtoolkit.surrogate.workflows.evaluator`) makes explicit, wherever
  a report is built, whether a study's repeated evaluations came from
  real FEA or a surrogate -- a Monte Carlo study is never silently
  pointed at a surrogate.
- **Version 33.** `Evaluator` (`HighFidelityEvaluator`/`SurrogateEvaluator`)
  is the shared interface an optimization algorithm could evaluate a
  design through. This version provides only the *architecture* --
  `OptimizationRunner`/the Version 33 algorithms are not yet wired to
  accept an `Evaluator` in place of the direct FEA call. Every
  `DesignEvaluation` a `SurrogateEvaluator` returns carries
  `metadata["surrogate_derived"] = True` plus the originating
  model/dataset identity and domain status, so it can never be mistaken
  for a verified result.

## High-fidelity verification

`verify_against_high_fidelity(model, base_project, design_points, response_extractors, tolerances=...)`
(`femtoolkit.surrogate.workflows.verification`) both predicts and
actually simulates every design point, returning one
`SurrogateVerificationRecord` per point with an `AcceptanceStatus`
(`ACCEPT`/`REVIEW`/`REJECT`/`FAILED`). `REVIEW` covers a verified result
with no tolerance configured, or one at/outside the training domain --
a human should look at it rather than treating it as automatically
acceptable. This is the mechanism behind
`examples/surrogate/surrogate_assisted_optimization.py`'s final step: a
surrogate-selected candidate is never reported as a verified engineering
result until this check has actually run.

## Persistence and reproducibility

`femtoolkit.surrogate.persistence` saves a surrogate model as plain
JSON (`save_surrogate_model`/`load_surrogate_model`) and a POD model as
JSON plus a NumPy `.npz` archive (`save_rom_model`/`load_rom_model`) --
never `pickle`. Loading validates both the on-disk schema version and
required fields before trusting the file, raising `ModelPersistenceError`
(missing/corrupted file) or `IncompatibleModelVersionError` (schema
mismatch) rather than silently deserializing something unexpected.
Every trained model records its training dataset ID/version, feature
and response names, random seed, model version, and the toolkit's
software version (`TrainingMetadata`) -- `PolynomialRegressionSurrogate`
and `RBFSurrogate` both use no randomness during fitting itself, so the
same dataset and configuration always reproduce the same model.

## Scope boundary (this version)

No neural networks, no deep learning, no Gaussian process regression,
no Bayesian optimization, no advanced active learning, no
topology/shape optimization, no adjoint methods, no distributed/GPU
surrogate training, and no formal uncertainty quantification of a
surrogate's own prediction error. See the
[Version 36 preview](../README.md#roadmap) for what builds on this
foundation next.
