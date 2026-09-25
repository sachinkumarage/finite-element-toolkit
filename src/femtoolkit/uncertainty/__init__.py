"""Uncertainty quantification and robust engineering analysis (Version 31).

Answers the question Version 30 left deterministic: *how does
uncertainty in the inputs affect uncertainty in the results?* Real
engineering inputs (material properties, loads, dimensions, boundary
conditions) are rarely known exactly. This package represents such a
parameter as a random variable
(:mod:`femtoolkit.uncertainty.distributions`,
:mod:`femtoolkit.uncertainty.parameters`), draws samples from it
(:mod:`femtoolkit.uncertainty.sampling`), runs a Monte Carlo study over
those samples by reusing the Version 30 scenario/run infrastructure
directly -- no second simulation execution system --
(:mod:`femtoolkit.uncertainty.monte_carlo`), and turns the resulting
output samples into statistics, confidence intervals, correlation, and
an empirical limit-exceedance estimate
(:mod:`femtoolkit.uncertainty.statistics`,
:mod:`femtoolkit.uncertainty.confidence`,
:mod:`femtoolkit.uncertainty.correlation`,
:mod:`femtoolkit.uncertainty.reliability`). See ``docs/uncertainty.md``
for the full guide.

**Explicit scope exclusions (by design, this version).** No machine
learning, no surrogate models, no Bayesian optimization, no topology or
gradient-based optimization, no genetic algorithms, no rigorous
reliability methods (FORM/SORM, importance sampling), no polynomial
chaos expansion, no Gaussian processes, no distributed/GPU Monte Carlo.
Those are explicitly future scope -- see the Version 32 preview in the
main README.
"""

from __future__ import annotations

from femtoolkit.uncertainty.confidence import ConfidenceInterval, confidence_interval_mean
from femtoolkit.uncertainty.correlation import (
    CorrelationResult,
    correlation_summary,
    pearson_correlation,
    spearman_correlation,
)
from femtoolkit.uncertainty.distributions import (
    DeterministicDistribution,
    Distribution,
    LognormalDistribution,
    NormalDistribution,
    UniformDistribution,
    distribution_from_dict,
    distribution_to_dict,
)
from femtoolkit.uncertainty.monte_carlo import (
    DEFAULT_MAX_SAMPLES,
    InvalidSampleRecord,
    MonteCarloConfig,
    MonteCarloResult,
    MonteCarloRunner,
)
from femtoolkit.uncertainty.parameters import UncertainParameter, UncertaintyCategory
from femtoolkit.uncertainty.plots import plot_input_distribution, plot_output_histogram
from femtoolkit.uncertainty.reliability import ExceedanceResult, exceedance_probability
from femtoolkit.uncertainty.report import (
    UncertaintyReport,
    build_uncertainty_report,
    render_uncertainty_report_html,
    render_uncertainty_report_markdown,
    save_uncertainty_report,
)
from femtoolkit.uncertainty.sampling import (
    LATIN_HYPERCUBE_METHOD,
    RANDOM_METHOD,
    SAMPLING_METHODS,
    SampleSet,
    generate_latin_hypercube_samples,
    generate_random_samples,
    generate_samples,
)
from femtoolkit.uncertainty.statistics import (
    DEFAULT_PERCENTILES,
    ConvergencePoint,
    OutputStatistics,
    compute_convergence_series,
    compute_output_statistics,
)

__all__ = [
    "DEFAULT_MAX_SAMPLES",
    "DEFAULT_PERCENTILES",
    "LATIN_HYPERCUBE_METHOD",
    "RANDOM_METHOD",
    "SAMPLING_METHODS",
    "ConfidenceInterval",
    "ConvergencePoint",
    "CorrelationResult",
    "DeterministicDistribution",
    "Distribution",
    "ExceedanceResult",
    "InvalidSampleRecord",
    "LognormalDistribution",
    "MonteCarloConfig",
    "MonteCarloResult",
    "MonteCarloRunner",
    "NormalDistribution",
    "OutputStatistics",
    "SampleSet",
    "UncertainParameter",
    "UncertaintyCategory",
    "UncertaintyReport",
    "UniformDistribution",
    "build_uncertainty_report",
    "compute_convergence_series",
    "compute_output_statistics",
    "confidence_interval_mean",
    "correlation_summary",
    "distribution_from_dict",
    "distribution_to_dict",
    "exceedance_probability",
    "generate_latin_hypercube_samples",
    "generate_random_samples",
    "generate_samples",
    "pearson_correlation",
    "plot_input_distribution",
    "plot_output_histogram",
    "render_uncertainty_report_html",
    "render_uncertainty_report_markdown",
    "save_uncertainty_report",
    "spearman_correlation",
]
