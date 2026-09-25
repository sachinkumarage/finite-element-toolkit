"""Probability distributions for uncertain engineering parameters (Version 31).

**Deterministic vs. uncertain inputs.** A deterministic analysis assumes
every input is one exact value, ``y = f(x)``. A real engineering input
is often better represented as a random variable, ``X``, so the
simulation output becomes uncertain too: ``Y = f(X)``. This module
provides the small, closed set of distributions this toolkit uses to
represent that random variable -- no distribution-fitting, no arbitrary
user-supplied density, just the handful of shapes that cover the
overwhelming majority of practical engineering uncertainty.

**Aleatory vs. epistemic uncertainty.** Aleatory uncertainty is natural,
irreducible variability (e.g. manufactured material strength varying
from sample to sample); epistemic uncertainty is a lack of knowledge
(e.g. an incompletely characterized boundary-condition stiffness). This
module does not need to distinguish the two mathematically -- both are
represented by a distribution the same way -- but
:class:`~femtoolkit.uncertainty.parameters.UncertainParameter` records
which category a given parameter represents, so a report can say so.

Every distribution supports the same small interface: draw samples,
report its mean/standard deviation, report physical bounds (``None`` on
a side with no hard bound), and support the percent-point function
(``ppf``, the inverse CDF) -- the operation
:mod:`femtoolkit.uncertainty.sampling`'s Latin Hypercube sampler needs
to turn a stratified ``[0, 1)`` quantile into an actual sample.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from femtoolkit.exceptions import ValidationError


class Distribution(ABC):
    """A scalar probability distribution for one uncertain parameter."""

    @abstractmethod
    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draw ``size`` independent and identically distributed samples.

        Args:
            rng: The random generator to draw from (see
                :mod:`femtoolkit.uncertainty.sampling` for how a study's
                seed produces this generator).
            size: How many samples to draw.

        Returns:
            A 1D array of ``size`` samples.
        """

    @abstractmethod
    def ppf(self, quantiles: np.ndarray) -> np.ndarray:
        """The percent-point function (inverse CDF): quantile -> value.

        Args:
            quantiles: Values in ``[0, 1]``.

        Returns:
            The distribution value at each quantile, same shape as
            ``quantiles``.
        """

    @abstractmethod
    def mean(self) -> float:
        """This distribution's mean."""

    @abstractmethod
    def std(self) -> float:
        """This distribution's standard deviation (``0.0`` for a deterministic value)."""

    def bounds(self) -> tuple[float | None, float | None]:
        """This distribution's hard support bounds, ``(lower, upper)``.

        ``None`` on a side means unbounded on that side. The default
        (unbounded both sides) is overridden by distributions with
        inherent support limits (e.g. a lognormal is never negative).
        """
        return (None, None)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """The probability density at ``x``.

        Raises:
            NotImplementedError: For a distribution with no density
                (a point mass, i.e. :class:`DeterministicDistribution`).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define a probability density function."
        )


@dataclass(frozen=True)
class DeterministicDistribution(Distribution):
    """A single, exact value with no variability -- ``X = c``.

    Useful so a parameter can be temporarily "turned off" (treated as
    certain) without removing it from an
    :class:`~femtoolkit.uncertainty.parameters.UncertainParameter` list,
    and as the natural degenerate case every other distribution should
    not be confused with.

    Attributes:
        value: The fixed value every sample equals.
    """

    value: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValidationError(
                f"DeterministicDistribution value must be finite, got {self.value!r}."
            )

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        del rng
        return np.full(size, self.value, dtype=float)

    def ppf(self, quantiles: np.ndarray) -> np.ndarray:
        quantiles = np.asarray(quantiles, dtype=float)
        if np.any((quantiles < 0.0) | (quantiles > 1.0)):
            raise ValidationError("Quantiles passed to ppf() must lie in [0, 1].")
        return np.full(quantiles.shape, self.value, dtype=float)

    def mean(self) -> float:
        return self.value

    def std(self) -> float:
        return 0.0

    def bounds(self) -> tuple[float | None, float | None]:
        return (self.value, self.value)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "DeterministicDistribution has no probability density (a point mass); use mean()."
        )


@dataclass(frozen=True)
class UniformDistribution(Distribution):
    """A uniform distribution, ``X ~ U(low, high)``.

    Attributes:
        low: The lower bound (inclusive).
        high: The upper bound (inclusive), must exceed ``low``.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.low) and math.isfinite(self.high)):
            raise ValidationError("UniformDistribution bounds must be finite.")
        if not self.low < self.high:
            raise ValidationError(
                f"UniformDistribution requires low < high (got low={self.low}, high={self.high})."
            )

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.uniform(self.low, self.high, size=size)

    def ppf(self, quantiles: np.ndarray) -> np.ndarray:
        q = np.clip(np.asarray(quantiles, dtype=float), 0.0, 1.0)
        return self.low + q * (self.high - self.low)

    def mean(self) -> float:
        return (self.low + self.high) / 2.0

    def std(self) -> float:
        return (self.high - self.low) / math.sqrt(12.0)

    def bounds(self) -> tuple[float | None, float | None]:
        return (self.low, self.high)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        density = 1.0 / (self.high - self.low)
        return np.where((x >= self.low) & (x <= self.high), density, 0.0)


@dataclass(frozen=True)
class NormalDistribution(Distribution):
    """A normal (Gaussian) distribution, ``X ~ N(mean, std)``.

    Unbounded on both sides -- a physical lower bound (e.g. ``E > 0``)
    is enforced separately by
    :class:`~femtoolkit.uncertainty.parameters.UncertainParameter`, not
    by this distribution itself (a normal has no natural hard bound).

    Attributes:
        mean_value: The distribution mean.
        std_value: The distribution standard deviation, must be positive.
    """

    mean_value: float
    std_value: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean_value):
            raise ValidationError("NormalDistribution mean must be finite.")
        if not (math.isfinite(self.std_value) and self.std_value > 0.0):
            raise ValidationError("NormalDistribution standard deviation must be positive.")

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.normal(self.mean_value, self.std_value, size=size)

    def ppf(self, quantiles: np.ndarray) -> np.ndarray:
        from scipy.stats import norm

        return norm.ppf(quantiles, loc=self.mean_value, scale=self.std_value)

    def mean(self) -> float:
        return self.mean_value

    def std(self) -> float:
        return self.std_value

    def pdf(self, x: np.ndarray) -> np.ndarray:
        from scipy.stats import norm

        return norm.pdf(x, loc=self.mean_value, scale=self.std_value)


@dataclass(frozen=True)
class LognormalDistribution(Distribution):
    """A lognormal distribution for a strictly positive quantity, ``ln(X) ~ N(mu, sigma)``.

    Parameterized by the mean/standard deviation of the underlying
    *log-space* normal (``mu``/``sigma``), the standard mathematical
    parameterization -- not the mean/standard deviation of ``X`` itself
    (those are derived; see :meth:`mean`/:meth:`std`, and
    :meth:`from_mean_std` to construct from them directly).

    Attributes:
        mu: The mean of ``ln(X)``.
        sigma: The standard deviation of ``ln(X)``, must be positive.
    """

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mu):
            raise ValidationError("LognormalDistribution mu must be finite.")
        if not (math.isfinite(self.sigma) and self.sigma > 0.0):
            raise ValidationError("LognormalDistribution sigma must be positive.")

    @classmethod
    def from_mean_std(cls, mean: float, std: float) -> LognormalDistribution:
        """Construct from the desired mean/standard deviation of ``X`` itself.

        Args:
            mean: The desired mean of ``X`` (must be positive).
            std: The desired standard deviation of ``X`` (must be positive).

        Returns:
            A :class:`LognormalDistribution` whose ``mean()``/``std()``
            match the requested values.
        """
        if not (math.isfinite(mean) and mean > 0.0):
            raise ValidationError("Lognormal mean must be positive.")
        if not (math.isfinite(std) and std > 0.0):
            raise ValidationError("Lognormal standard deviation must be positive.")
        variance_ratio = (std / mean) ** 2
        sigma = math.sqrt(math.log(1.0 + variance_ratio))
        mu = math.log(mean) - sigma**2 / 2.0
        return cls(mu=mu, sigma=sigma)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.lognormal(self.mu, self.sigma, size=size)

    def ppf(self, quantiles: np.ndarray) -> np.ndarray:
        from scipy.stats import lognorm

        return lognorm.ppf(quantiles, s=self.sigma, scale=math.exp(self.mu))

    def mean(self) -> float:
        return math.exp(self.mu + self.sigma**2 / 2.0)

    def std(self) -> float:
        variance = (math.exp(self.sigma**2) - 1.0) * math.exp(2.0 * self.mu + self.sigma**2)
        return math.sqrt(variance)

    def bounds(self) -> tuple[float | None, float | None]:
        return (0.0, None)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        from scipy.stats import lognorm

        return lognorm.pdf(x, s=self.sigma, scale=math.exp(self.mu))


_DISTRIBUTION_TYPES: dict[str, type[Distribution]] = {
    "deterministic": DeterministicDistribution,
    "uniform": UniformDistribution,
    "normal": NormalDistribution,
    "lognormal": LognormalDistribution,
}


def distribution_to_dict(distribution: Distribution) -> dict:
    """Return a plain, JSON-serializable representation of a distribution.

    Args:
        distribution: The distribution to serialize.

    Returns:
        A dict with a ``"type"`` discriminator plus that distribution's
        own fields, suitable for :func:`distribution_from_dict`.
    """
    for type_name, cls in _DISTRIBUTION_TYPES.items():
        if isinstance(distribution, cls):
            fields = {
                key: value
                for key, value in vars(distribution).items()
                if not key.startswith("_")
            }
            return {"type": type_name, **fields}
    raise ValidationError(f"Unknown distribution type: {type(distribution).__name__!r}.")


def distribution_from_dict(data: dict) -> Distribution:
    """Reconstruct a :class:`Distribution` from :func:`distribution_to_dict`'s output.

    Args:
        data: A dict shaped like :func:`distribution_to_dict`'s output.

    Returns:
        The matching :class:`Distribution` subclass instance.

    Raises:
        ValidationError: If ``data["type"]`` is not a known distribution type.
    """
    data = dict(data)
    type_name = data.pop("type", None)
    cls = _DISTRIBUTION_TYPES.get(type_name)
    if cls is None:
        raise ValidationError(
            f"Unknown distribution type {type_name!r}; known types: "
            f"{sorted(_DISTRIBUTION_TYPES)}."
        )
    return cls(**data)


__all__ = [
    "DeterministicDistribution",
    "Distribution",
    "LognormalDistribution",
    "NormalDistribution",
    "UniformDistribution",
    "distribution_from_dict",
    "distribution_to_dict",
]
