# Uncertainty Quantification & Robust Engineering Analysis (Version 31)

Answers the question Version 30 left deterministic: *how does
uncertainty in the inputs affect uncertainty in the results?* This
document covers probability distributions, sampling, Monte Carlo
studies, statistical output analysis, confidence intervals, correlation
and sensitivity, and the reliability foundation. See
[`docs/studies.md`](studies.md) for the Version 30 parameter-study
framework this version builds on, and the main
[README](../README.md#version-31) for a shorter overview.

## Deterministic vs. uncertain inputs

A deterministic simulation assumes every input is one exact value,
``y = f(x)``. A real engineering input is often better represented as a
random variable, ``X``, so the simulation output becomes uncertain too:
``Y = f(X)``. For example, ``E ~ N(mu_E, sigma_E)`` (Young's modulus,
normally distributed around a mean with some standard deviation) makes
tip displacement ``u = f(P, E, L, A)`` a random variable as well.

## Aleatory vs. epistemic uncertainty

- **Aleatory** uncertainty is natural, irreducible variability -- e.g.
  sample-to-sample variation in a manufactured material's strength.
- **Epistemic** uncertainty comes from a lack of knowledge -- e.g. an
  incompletely characterized boundary-condition stiffness. More
  measurement or testing could, in principle, reduce it.

`UncertainParameter.category` (`UncertaintyCategory.ALEATORY` /
`.EPISTEMIC`) records which one a parameter represents, purely for
labeling a report -- both are sampled from a distribution the same way.

## Probability distributions

`femtoolkit.uncertainty.distributions` provides four distributions
behind one small interface (`sample`, `ppf`, `mean`, `std`, `bounds`,
`pdf`):

- `DeterministicDistribution(value)` -- `X = c`, a point mass with no
  density (has no `pdf`).
- `UniformDistribution(low, high)` -- `X ~ U(low, high)`.
- `NormalDistribution(mean_value, std_value)` -- `X ~ N(mu, sigma)`,
  unbounded.
- `LognormalDistribution(mu, sigma)` -- `ln(X) ~ N(mu, sigma)`, always
  positive; `LognormalDistribution.from_mean_std(mean, std)` constructs
  one from the desired mean/std of `X` itself (the more common way an
  engineer specifies it) rather than the log-space parameters.

Built on NumPy/SciPy (`scipy.stats.norm`/`lognorm` for `ppf`/`pdf`) --
both already core dependencies since earlier versions; no new
statistical library was added.

```python
from femtoolkit.uncertainty import NormalDistribution

youngs_modulus = NormalDistribution(mean_value=200e9, std_value=5e9)
youngs_modulus.mean()   # 200e9
youngs_modulus.std()    # 5e9
```

## Uncertain parameters and physical validity

`UncertainParameter` pairs a distribution with the same dotted override
path `Scenario`/`ParameterDefinition` (Version 30) already use to
identify a project field:

```python
from femtoolkit.uncertainty import UncertainParameter, UncertaintyCategory, NormalDistribution

youngs_modulus = UncertainParameter(
    path="material.youngs_modulus",
    label="Young's Modulus",
    distribution=NormalDistribution(200e9, 5e9),
    units="Pa",
    category=UncertaintyCategory.ALEATORY,
    physical_lower_bound=0.0,   # E > 0
)
```

`effective_bounds()` combines the distribution's own bounds (e.g. a
lognormal is never negative) with `physical_lower_bound`/
`physical_upper_bound` on each side, taking the tighter of the two.
`is_physically_valid(value)` checks a sampled value against those
combined bounds -- a normal distribution can mathematically produce a
negative Young's modulus in its tail; that sample is caught and never
silently accepted.

## Sampling

```python
from femtoolkit.uncertainty import generate_samples

sample_set = generate_samples([youngs_modulus], n_samples=100, method="random", seed=42)
```

The same seed always reproduces the same sample sequence
(`numpy.random.default_rng(seed)`, never global random state). Two
methods:

- **`"random"`** -- each parameter sampled independently from its own
  distribution.
- **`"latin_hypercube"`** -- for `N` samples, each parameter's
  probability range is divided into `N` equal-probability intervals and
  exactly one sample is drawn from each (a random point within the
  interval, then a random permutation of interval-to-sample assignment,
  independently per parameter). This guarantees every part of a
  parameter's range is represented at the same sample count as random
  sampling -- random sampling can by chance cluster or leave gaps for
  a given draw, LHS cannot. It does not guarantee joint coverage of
  every input *combination*, and this toolkit's implementation is
  deliberately basic stratification, not a full quasi-Monte Carlo
  method.

## Monte Carlo studies

```text
Uncertain Inputs -> Sampling -> Scenario generation -> FEA runs -> Output collection -> Statistics
```

```python
from femtoolkit.uncertainty import MonteCarloConfig, MonteCarloRunner

config = MonteCarloConfig(
    study_id="material-study", name="Material Uncertainty Study",
    base_project=base_project, parameters=[youngs_modulus],
    output_quantities=["maximum_displacement"],
    n_samples=200, seed=42, method="random",
)
result = MonteCarloRunner().run(config)
```

**No second simulation execution system.** Each sample becomes a
`Scenario` (Version 30) with `parameter_overrides` set from its sampled
values, and execution is delegated entirely to `StudyRunner` (or, for
`fail_fast=True`, directly to the same `SimulationRunManager` that
`StudyRunner` itself uses internally). `femtoolkit.uncertainty` performs
no FEA computation and no scenario/run bookkeeping of its own.

**Safety before execution.** `MonteCarloConfig.__post_init__` rejects
`n_samples > max_samples` (default 1000) immediately, reusing Version
30's `StudySizeExceededError` and its exact "stop before execution,
explain how to reduce" pattern -- a study is never silently reduced.

**Failure handling.** Every sample ends up in one of three states:

- **Rejected as physically invalid** -- caught by `UncertainParameter.is_physically_valid`
  *before* a scenario is even built; recorded as an `InvalidSampleRecord`
  (sample index, drawn values, reason), never sent to the solver.
- **Failed** -- the scenario was valid to sample but failed project
  validation or the solver itself; recorded as a `FAILED`
  `SimulationRun` with a structured `error_stage`/`error_message`
  (identical to Version 30).
- **Successful** -- a completed `SimulationRun`.

`fail_fast=True` stops execution at the first failed (not invalid)
sample; `fail_fast=False` (default) runs every valid sample regardless
of earlier failures, and `MonteCarloResult.n_failed` always reports how
many failed.

## Output statistics

```python
from femtoolkit.studies.extractors import get_extractor
from femtoolkit.uncertainty import compute_output_statistics

values = result.output_values(get_extractor("maximum_displacement"))
stats = compute_output_statistics(values, "Maximum displacement", units="m")
stats.mean, stats.std, stats.coefficient_of_variation, stats.percentiles
```

Every statistic that needs at least two samples (`std`, `variance`,
`coefficient_of_variation`) is `None` rather than a fabricated number
when there are fewer than two successful runs. Percentiles are
configurable (default 5/25/50/75/95) and describe *this sampled output
distribution* -- see the next section for the different question a
confidence interval answers.

`compute_convergence_series(values)` traces the running mean and
standard error as samples accumulate in draw order -- a simple
diagnostic for whether an estimate has visibly settled, since finishing
the simulation runs alone is not evidence of statistical convergence.

## Confidence intervals vs. percentiles -- not the same statement

A **percentile** (P95, say) describes the sampled output distribution:
"95% of this sample's successful runs fall below this value." A
**confidence interval** describes how precisely the *mean* has been
estimated from a finite sample: "if this study were repeated many
times, 95% of the resulting intervals would contain the true population
mean." These are different statistical statements, and this toolkit
never calls one the other.

```python
from femtoolkit.uncertainty import confidence_interval_mean

ci = confidence_interval_mean(values, "Maximum displacement", confidence_level=0.95)
ci.lower, ci.mean, ci.upper
```

Computed with the Student-*t* distribution (`mean +/- t(alpha/2, n-1) *
standard_error`), appropriate since the population standard deviation
is never actually known here, only estimated from the same finite
sample -- the situation the *t* distribution is built for. As the
sample count grows, *t* converges to the normal (*z*) approximation, so
this is never a worse choice, only ever a more appropriate one for
small samples.

## Correlation: global, sample-based, descriptive

```python
from femtoolkit.uncertainty import correlation_summary

x, y = result.successful_pairs("material.youngs_modulus", extractor)
summary = correlation_summary("Maximum displacement", {"material.youngs_modulus": ("Young's Modulus", x, y)})
```

- **Pearson's r** measures the strength of a *linear* association.
- **Spearman's rho** captures monotonic relationships that are not
  necessarily linear (e.g. `y = x**5` gives Spearman `rho = 1.0` but a
  Pearson `r < 1.0`).

**Correlation is not causation**, and a low Pearson `r` does not mean a
parameter has no effect -- it may have a strong nonlinear or
non-monotonic one that Spearman's `rho` would reveal instead.
`correlation_summary` sorts by descending `|pearson_r|` as a
*descriptive statistical ranking* for a report table, never as an
engineering-importance ranking.

## Local sensitivity vs. global uncertainty analysis

Two different, complementary questions:

- **Local sensitivity** (Version 30, `femtoolkit.studies.sensitivity`):
  `S = (dy/y) / (dp/p)`, a finite-difference ratio between exactly two
  nearby evaluations around one nominal operating point.
- **Global uncertainty analysis** (this version): inputs vary over
  their full distributions; a Monte Carlo study estimates the output's
  behavior across that entire specified input-uncertainty space, not
  just near one point.

Neither supersedes the other. A study report's "Sensitivity
Information" section can include both side by side (see
`UncertaintyReport.local_sensitivities`) for direct comparison.

## Reliability foundation: empirical exceedance, not a reliability index

For a limit state `g(X) = R(X) - S(X)` (resistance minus demand),
failure is conventionally `g(X) < 0`. This toolkit supports the common
simplified case of one scalar output compared against a fixed
threshold:

```python
from femtoolkit.uncertainty import exceedance_probability

result_ = exceedance_probability(values, threshold=0.005, quantity_label="Maximum displacement")
result_.n_exceeding, result_.n_samples, result_.exceedance_frequency
```

`exceedance_frequency = n_exceeding / n_samples` is an **empirical
frequency from this specific Monte Carlo sample** -- not a rigorous
reliability index. No FORM/SORM, importance sampling, or subset
simulation is implemented in this version; a small sample carries
substantial sampling uncertainty of its own, especially for a rare
event (a small exceedance count). `ExceedanceResult` always reports the
underlying sample count so this limitation stays visible.

## Robustness metrics

No single "robustness score" is computed anywhere in this toolkit.
Robustness is represented transparently by the metrics already
described above: mean, standard deviation, coefficient of variation,
percentile spread, and limit-exceedance frequency -- reported
side by side, left for an engineer to weigh against real requirements.

## Uncertainty reports

`femtoolkit.uncertainty.report` extends the Version 30/29 reporting
approach to a nineteen-section document (kept out of
`femtoolkit.studies`/`femtoolkit.reporting` to preserve a one-way
uncertainty -> studies -> reporting dependency direction):

1. Study Summary
2. Base Model
3. Uncertain Parameters
4. Probability Distributions
5. Sampling Method
6. Sample Count
7. Random Seed
8. Simulation Run Statistics
9. Output Statistics
10. Percentiles
11. Confidence Intervals
12. Correlation Analysis
13. Limit Exceedance
14. Histograms
15. Sensitivity Information
16. Verification Status
17. Validation Status
18. Reproducibility Metadata
19. Limitations

```python
from femtoolkit.uncertainty import build_uncertainty_report, save_uncertainty_report

report = build_uncertainty_report(
    title="Material Uncertainty Study Report",
    study_summary="...", base_model_description="...",
    result=result, output_statistics=[stats], confidence_intervals=[ci],
    correlations=summary, conclusions="...",
)
save_uncertainty_report(report, "report.md", "markdown")
```

As with every prior reporting module in this toolkit, `conclusions` is
free text the caller supplies -- never generated automatically.

## Reproducibility

Every `MonteCarloResult` carries what is needed to reproduce a study
"subject to deterministic numerical execution" (floating-point behavior
can still vary slightly across software/hardware environments, even
with an identical seed):

- `config` -- the full `MonteCarloConfig` (study ID, sample count,
  seed, sampling method, every `UncertainParameter`/distribution).
- `sample_set` -- every sampled input value, valid and invalid alike.
- Each successful run's `reproducibility_metadata` (Version 29, toolkit/
  Python/dependency versions, model/solver configuration) and immutable
  `configuration_snapshot` (Version 30).

No duplicate FEA result-array storage is introduced: `MonteCarloResult`
reuses the Version 30 `StudyResult`/`SimulationRun` types directly
rather than persisting a second, parallel results format.

## GUI integration

A new Uncertainty Analysis page: define uncertain parameters and their
distributions, preview an input distribution before running anything,
configure and run a Monte Carlo study (sample count, seed, method,
`fail_fast`, output quantities), and review output statistics,
percentiles, a confidence interval, a histogram, correlation, limit
exceedance, and a downloadable report -- all through the layers
described above, with no distribution/sampling/Monte Carlo/statistical
logic implemented in the GUI page itself.

## No "best design" ranking

Nothing in `femtoolkit.uncertainty` synthesizes an overall robustness
score, a "safest" configuration, or a ranking of design alternatives --
matching this toolkit's existing Version 30 "no material ranking"
principle. Statistics, correlation, and exceedance frequency are
reported as computed; interpreting them against real requirements is
the engineer's responsibility.

## Explicit scope exclusions (this version)

No machine learning, no surrogate models, no Bayesian optimization, no
topology or gradient-based optimization, no genetic algorithms, no
rigorous reliability methods (FORM/SORM, importance sampling, subset
simulation), no polynomial chaos expansion, no Gaussian processes, no
distributed/GPU Monte Carlo, no CFD, and no new finite element,
material model, or solver algorithm. See the Version 32 preview in the
main README for the planned next direction (optimization and design
exploration, building on this version's infrastructure).

## Limitations

- **No distributed or parallel Monte Carlo execution.** Every sample
  runs sequentially (via the Version 30 `StudyRunner`); the loop's
  structure does not preclude wiring in the Version 27 parallel
  infrastructure later, but nothing is parallelized here.
- **Empirical exceedance, not a reliability index** (see above).
- **No probability distribution beyond the four listed** -- no
  triangular, beta, or user-supplied arbitrary distribution.
- **Correlation is not causation, and Pearson/Spearman do not capture
  every possible relationship shape.**
- **A confidence interval assumes a reasonably well-behaved sample
  mean** (the Student-*t* interval is asymptotically justified; for a
  very small or heavily skewed sample, treat it as approximate).
