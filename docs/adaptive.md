# Surrogate-Assisted Adaptive Optimization (Version 36)

Answers the question Version 35 left to a human: *once a surrogate exists, how
should it actually guide a design search, and when should it be trusted enough to
stop asking for another expensive FEA solve?* This document covers the adaptive
loop, candidate generation, the four sampling strategies, the trust-region
foundation, high-fidelity verification, and the Adaptive Optimization GUI page
(`femtoolkit.gui`, as of Version 37.1). See the main
[README](../README.md#version-36) for a shorter overview.

## The core loop

```text
Initial Engineering Samples -> High-Fidelity FEA -> Surrogate Model ->
Candidate Search -> Candidate Designs -> High-Fidelity Verification ->
Add New Sample -> Retrain Surrogate -> Repeat
```

For a design vector `x = (x_1, ..., x_n)`, the expensive simulation provides
`y = f(x)`; the Version 35 surrogate approximates `y_hat = f_hat(x) ~ f(x)`. The
underlying problem is `min_x f(x)` subject to `g_i(x) <= 0` and
`x_min <= x <= x_max` -- but every candidate `femtoolkit.adaptive` proposes is
scored against `f_hat`, never `f`, until it is explicitly verified with
`femtoolkit.surrogate.workflows.verification.verify_against_high_fidelity`. **The
surrogate accelerates the search; it never replaces the high-fidelity reference.**

## Candidate generation

`generate_candidate_pool(design_variables, n_candidates, seed)`
(`femtoolkit.adaptive.candidates`) draws a pool of random points directly from
each `DesignVariable.sample()` -- the same sampling a Version 32
`BoundedRandomSearch` candidate already uses. Every candidate is then evaluated
by the surrogate, never the high-fidelity model, via
`femtoolkit.surrogate.workflows.evaluator.SurrogateEvaluator`.

## Adaptive sampling: four strategies

`rank_candidates(...)` (`femtoolkit.adaptive.sampling`) scores a candidate pool
with one of four deliberately transparent heuristics (`SamplingStrategy`):

- **`DISTANCE`** (Strategy A, pure exploration) -- prefer points far, in
  normalized design space, from every existing training sample.
- **`ERROR`** (Strategy B, error-based refinement) -- prefer points with a worse
  applicability-domain status (`femtoolkit.surrogate.domain.DomainStatus`), used
  as a transparent proxy for "the surrogate is less confident here."
- **`OBJECTIVE`** (Strategy C, pure exploitation) -- prefer points the surrogate
  predicts are the best designs, ranked with the same feasibility-first
  comparison (`is_better_evaluation`) every optimization algorithm in this
  toolkit already uses.
- **`HYBRID`** (Strategy D) -- a weighted combination,

  ```text
  Score(x) = w_e * E(x) + w_o * O(x)
  ```

  where `E(x)` is the normalized exploration score and `O(x)` is the normalized
  objective score. This is an engineering heuristic, not a mathematically
  universal acquisition function -- there is no claim this finds a global
  optimum.

**Normalized design-space distance.** Every distance calculation
(`normalized_distance`) first rescales each dimension to `z_i = (x_i - x_min) /
(x_max - x_min)`, so a variable with a large physical range (a load in Newtons)
cannot dominate one with a small range (a thickness in meters) purely because of
unit choice. A zero-width range contributes nothing to the distance.

## Trust region (simple foundation)

`TrustRegion` (`femtoolkit.adaptive.trust_region`) is the ball `||x - x_c|| <=
Delta` around a design center, measured with the same normalized distance.
`expand()`/`contract()` grow/shrink the radius by a configurable factor, bounded
by `min_radius`/`max_radius`. This is a foundation, not a full trust-region
optimization algorithm: a caller decides when to call `expand`/`contract`,
typically from `prediction_agreement`'s ratio (see below).

## Prediction agreement

```text
rho = (f(x_c) - f(x_new)) / (f_hat(x_c) - f_hat(x_new))
```

(sign-flipped for maximization). `prediction_agreement` is an engineering
diagnostic only -- not a formal optimality criterion -- for whether the surrogate
is behaving reliably near the current design center. A near-zero predicted
improvement returns `None` rather than a numerically unstable ratio.

## Adaptive refinement

`run_refinement_step` (`femtoolkit.adaptive.refinement`) is one iteration: rank a
fresh candidate pool, verify the top-ranked candidate against real FEA, and -- if
that run completed -- add the new sample to the dataset
(`SnapshotDataset.with_additional_snapshots`) and retrain the surrogate
(`train_surrogate`). Every step reports a `SurrogateAcceptanceState`:

```text
SURROGATE_ONLY        -- only a prediction exists, not yet verified
PENDING_VERIFICATION  -- a verification has been requested but not completed
VERIFIED              -- the high-fidelity result confirms the prediction
                         within the configured error tolerance
VERIFICATION_FAILED   -- the high-fidelity run itself did not complete
REQUIRES_REFINEMENT   -- verified, but the surrogate disagreed beyond tolerance
```

A surrogate prediction is never implied to be equivalent to a verified FEA
result -- a good prediction for *one* response (e.g. a purely geometric mass) can
still coexist with `REQUIRES_REFINEMENT` if *another* response (e.g.
displacement) disagrees beyond tolerance.

## `AdaptiveStudy`

`AdaptiveStudy` (`femtoolkit.adaptive.study`) ties the whole loop together:
`generate_initial_samples` builds (or accepts a caller-supplied) initial dataset,
trains the first surrogate, then runs `run_refinement_step` up to
`RefinementConfig.max_iterations` times, stopping early on
`max_high_fidelity_evaluations` or a stalled improvement
(`improvement_tolerance`). The returned `AdaptiveStudyResult` separates:

- `best_verified_design`/`best_verified_objective` -- from real FEA only.
- `best_surrogate_predicted_design`/`best_surrogate_predicted_objective` -- the
  surrogate's own current belief.

A design is never reported as "optimal" -- only as the **best verified design
found** within the configured budget.

## Integration with Version 30/31/33/34/35

- **Version 30/34.** `generate_initial_samples` executes each initial point
  through the same `Scenario`/`apply_scenario` override mechanism and
  `evaluate_simulation_batch` every other high-fidelity batch in this toolkit
  uses -- optionally in parallel via an `OrchestrationConfig`.
- **Version 31.** An `Objective`/`Constraint` built from
  `femtoolkit.optimization.robust.robust_objective_statistic`/
  `robust_constraint_statistic` works with `AdaptiveStudy` exactly as it already
  does with Version 33 optimization -- no separate UQ integration was built.
- **Version 33.** `is_better_evaluation`/`feasibility_rank` rank every candidate
  exactly the way a Version 32/33 optimization algorithm already does.
- **Version 35.** Every dataset, scaler, surrogate model, validation metric, and
  verification record is the unmodified Version 35 implementation -- nothing is
  duplicated.

## The Adaptive Optimization GUI page

As of Version 37.1, this workflow is a page in the full engineering GUI
(`femtoolkit.gui`, `workflow_pages/adaptive_page.py`) rather than a separate
application -- the standalone quickstart app this section originally
described was merged into `femtoolkit.gui` in Version 37.1 and no longer
exists. The core library never imports Streamlit; the page only calls
`femtoolkit.adaptive`/`femtoolkit.application` APIs.

### Installing and running

```bash
pip install -e ".[gui]"
streamlit run streamlit_app.py
# or, equivalently:
streamlit run src/femtoolkit/gui/app.py
```

Then open the **Adaptive Optimization** page from the sidebar.

### Workflow

Unlike the former quickstart app's hardcoded demonstration project, the GUI
page operates on the *current* project (built on the Project/Material/Mesh/
Boundary Conditions/Loads pages): add one or more design variables (a dotted
override path, a name, and a continuous bound), choose an objective response
and an optional constraint from the same named extractors every other page
uses, configure the refinement loop, and run it. Results show the best
*verified* design, the final surrogate's validation metrics, a convergence
chart, and each iteration's surrogate prediction next to its high-fidelity
verification.

No expensive FEA runs automatically on a Streamlit rerun -- every action is
gated behind an explicit button ("Add Variable", "Run Adaptive Study").

### Deployment preparation

The GUI is ready for a Streamlit Community Cloud deployment: dependencies are
declared in `pyproject.toml` and mirrored in a root `requirements.txt`
(`.[gui]`, installing this package and Streamlit from the repository itself --
no absolute local paths), no secrets are read or stored, and every page starts
from repository files only. To deploy: push this repository to GitHub, create
a new Streamlit Community Cloud app pointing at it, set the main file path to
`streamlit_app.py`, and deploy.

## Limitations

No multi-fidelity modeling, co-kriging, Gaussian processes, deep learning, neural
networks, topology/shape/adjoint optimization, reinforcement learning,
distributed/GPU/cloud execution, digital twins, or formal reliability methods
(FORM/SORM). `rank_candidates` scores a caller-generated pool -- it does not
solve a continuous acquisition-function optimization itself. See the [Version 37
preview](../README.md#roadmap) for what builds on this foundation next.
