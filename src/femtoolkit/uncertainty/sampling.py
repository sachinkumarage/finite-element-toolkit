"""Sampling uncertain parameters: random and Latin Hypercube (Version 31).

Two sampling methods are supported, both producing the same
:class:`SampleSet` shape so a caller (in practice,
:mod:`femtoolkit.uncertainty.monte_carlo`) never needs to know which one
was used:

- **Random sampling** (:func:`generate_random_samples`) draws each
  parameter's samples independently from its own distribution -- the
  textbook Monte Carlo approach, simple and unbiased, but for a small
  sample count can leave gaps or clusters in the input space by chance.
- **Latin Hypercube Sampling** (:func:`generate_latin_hypercube_samples`)
  improves coverage for the same sample count: for ``N`` samples, each
  parameter's probability range is divided into ``N`` equal-probability
  intervals, and exactly one sample is drawn from each interval (in a
  random, independently-permuted order per parameter, so no artificial
  correlation between parameters is introduced). This guarantees the
  full range of each parameter is represented, at the same sample count
  and cost as random sampling -- it does not, however, guarantee joint
  coverage of every input *combination*, and it is deliberately kept to
  this basic stratified form rather than a more advanced quasi-Monte
  Carlo method.

Reproducibility is achieved the same way throughout this toolkit's
random-number use: a caller-supplied integer seed is fed to
``numpy.random.default_rng(seed)`` once, producing one
:class:`numpy.random.Generator` instance that is never reused as global
state -- the same seed always reproduces the same sample sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from femtoolkit.exceptions import ValidationError
from femtoolkit.uncertainty.parameters import UncertainParameter

RANDOM_METHOD = "random"
LATIN_HYPERCUBE_METHOD = "latin_hypercube"
SAMPLING_METHODS = (RANDOM_METHOD, LATIN_HYPERCUBE_METHOD)


@dataclass(frozen=True)
class SampleSet:
    """A table of sampled input values: one row per sample, one column per parameter.

    Attributes:
        parameter_paths: Each column's parameter path, in column order.
        values: A ``(n_samples, n_parameters)`` array of sampled values.
        seed: The random seed used to produce this sample set, or
            ``None`` if none was supplied (non-reproducible).
        method: ``"random"`` or ``"latin_hypercube"``.
    """

    parameter_paths: list[str]
    values: np.ndarray
    seed: int | None
    method: str

    @property
    def n_samples(self) -> int:
        """The number of sampled rows."""
        return int(self.values.shape[0])

    def row(self, index: int) -> dict[str, float]:
        """The sampled parameter values for one sample, keyed by parameter path.

        Args:
            index: The sample's row index.

        Returns:
            ``{parameter_path: value}`` for that sample.
        """
        return {
            path: float(self.values[index, column])
            for column, path in enumerate(self.parameter_paths)
        }

    def column(self, parameter_path: str) -> np.ndarray:
        """Every sample's value for one parameter.

        Args:
            parameter_path: The parameter's path (must be one of
                :attr:`parameter_paths`).

        Returns:
            A 1D array of length :attr:`n_samples`.

        Raises:
            ValidationError: If ``parameter_path`` is not in this sample set.
        """
        if parameter_path not in self.parameter_paths:
            raise ValidationError(
                f"Unknown parameter path {parameter_path!r}; sampled paths: "
                f"{self.parameter_paths}."
            )
        return self.values[:, self.parameter_paths.index(parameter_path)]


def _validate_sampling_inputs(parameters: list[UncertainParameter], n_samples: int) -> None:
    if not parameters:
        raise ValidationError("At least one UncertainParameter is required to sample.")
    if n_samples < 1:
        raise ValidationError(f"n_samples must be at least 1, got {n_samples}.")


def generate_random_samples(
    parameters: list[UncertainParameter], n_samples: int, seed: int | None = None
) -> SampleSet:
    """Draw independent random samples for every parameter.

    Args:
        parameters: The uncertain parameters to sample.
        n_samples: How many samples to draw.
        seed: A random seed for reproducibility; the same seed always
            produces the same sample set.

    Returns:
        A :class:`SampleSet` with ``method="random"``.
    """
    _validate_sampling_inputs(parameters, n_samples)
    rng = np.random.default_rng(seed)
    columns = [parameter.distribution.sample(rng, n_samples) for parameter in parameters]
    values = np.column_stack(columns)
    return SampleSet(
        parameter_paths=[parameter.path for parameter in parameters],
        values=values,
        seed=seed,
        method=RANDOM_METHOD,
    )


def generate_latin_hypercube_samples(
    parameters: list[UncertainParameter], n_samples: int, seed: int | None = None
) -> SampleSet:
    """Draw Latin Hypercube samples for every parameter.

    For each parameter independently: divide ``[0, 1)`` into
    ``n_samples`` equal intervals, draw one uniformly random point
    within each interval (stratification), then randomly permute the
    interval-to-sample assignment (independently per parameter, so
    parameters are not spuriously correlated). Each stratified quantile
    is then converted to an actual value via the parameter's
    :meth:`~femtoolkit.uncertainty.distributions.Distribution.ppf`.

    Args:
        parameters: The uncertain parameters to sample.
        n_samples: How many samples to draw (also the number of strata
            per parameter).
        seed: A random seed for reproducibility.

    Returns:
        A :class:`SampleSet` with ``method="latin_hypercube"``.
    """
    _validate_sampling_inputs(parameters, n_samples)
    rng = np.random.default_rng(seed)
    columns = []
    for parameter in parameters:
        interval_starts = np.arange(n_samples, dtype=float)
        jitter = rng.uniform(0.0, 1.0, size=n_samples)
        quantiles = (interval_starts + jitter) / n_samples
        rng.shuffle(quantiles)
        columns.append(parameter.distribution.ppf(quantiles))
    values = np.column_stack(columns)
    return SampleSet(
        parameter_paths=[parameter.path for parameter in parameters],
        values=values,
        seed=seed,
        method=LATIN_HYPERCUBE_METHOD,
    )


def generate_samples(
    parameters: list[UncertainParameter],
    n_samples: int,
    method: str = RANDOM_METHOD,
    seed: int | None = None,
) -> SampleSet:
    """Dispatch to :func:`generate_random_samples` or :func:`generate_latin_hypercube_samples`.

    Args:
        parameters: The uncertain parameters to sample.
        n_samples: How many samples to draw.
        method: ``"random"`` or ``"latin_hypercube"``.
        seed: A random seed for reproducibility.

    Returns:
        A :class:`SampleSet`.

    Raises:
        ValidationError: If ``method`` is not a known sampling method.
    """
    if method == RANDOM_METHOD:
        return generate_random_samples(parameters, n_samples, seed)
    if method == LATIN_HYPERCUBE_METHOD:
        return generate_latin_hypercube_samples(parameters, n_samples, seed)
    raise ValidationError(
        f"Unknown sampling method {method!r}; expected one of {SAMPLING_METHODS}."
    )


__all__ = [
    "LATIN_HYPERCUBE_METHOD",
    "RANDOM_METHOD",
    "SAMPLING_METHODS",
    "SampleSet",
    "generate_latin_hypercube_samples",
    "generate_random_samples",
    "generate_samples",
]
