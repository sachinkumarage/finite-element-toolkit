"""Input/output correlation and a descriptive sensitivity summary (Version 31).

**Correlation is not causation, and Pearson is not the whole story.**
Pearson's ``r`` measures the strength of a *linear* association between
one input parameter and one output quantity across the Monte Carlo
sample; a low Pearson ``r`` does not mean a parameter has no effect --
it may have a strong but nonlinear or non-monotonic effect. Spearman's
``rho`` (rank correlation) captures monotonic relationships that are not
necessarily linear, at the cost of not distinguishing the exact shape.
Neither establishes physical cause and effect on its own -- both are
purely descriptive statistics computed from the sample this study
happened to draw.

**This is global, sample-based sensitivity, distinct from
:mod:`femtoolkit.studies.sensitivity`'s local finite-difference
sensitivity.** The Version 30 ``S = (dy/y)/(dp/p)`` describes behavior
at one nominal point via two nearby evaluations; the correlation summary
here describes association across an entire sampled input distribution.
Neither one supersedes the other -- they answer different questions
(see ``docs/uncertainty.md``, "Local sensitivity vs. global
uncertainty").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from femtoolkit.exceptions import ValidationError


@dataclass(frozen=True)
class CorrelationResult:
    """The correlation between one input parameter and one output quantity.

    Attributes:
        parameter_path: The input parameter's dotted override path.
        parameter_label: The input parameter's display label.
        quantity_label: The output quantity's display label.
        pearson_r: The Pearson linear correlation coefficient, in ``[-1, 1]``.
        spearman_rho: The Spearman rank correlation coefficient, in
            ``[-1, 1]``, or ``None`` if not computed.
        n_samples: How many paired (input, output) samples the
            correlation was computed from.
    """

    parameter_path: str
    parameter_label: str
    quantity_label: str
    pearson_r: float
    spearman_rho: float | None
    n_samples: int


def _validate_pair(x: np.ndarray, y: np.ndarray) -> None:
    if x.shape != y.shape:
        raise ValidationError(
            f"Input and output sample arrays must be the same shape (got {x.shape} and {y.shape})."
        )
    if x.size < 2:
        raise ValidationError(f"Correlation requires at least 2 paired samples, got {x.size}.")


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Pearson linear correlation coefficient between two equal-length samples.

    Args:
        x: One parameter's sampled input values.
        y: The corresponding output values, same length and pairing as ``x``.

    Returns:
        Pearson's ``r``, in ``[-1, 1]``.

    Raises:
        ValidationError: If the arrays differ in length, have fewer
            than 2 paired samples, or either has zero variance (a
            correlation coefficient is undefined against a constant).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    _validate_pair(x, y)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        raise ValidationError("Pearson correlation is undefined when a sample has zero variance.")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Spearman rank correlation coefficient between two equal-length samples.

    Args:
        x: One parameter's sampled input values.
        y: The corresponding output values, same length and pairing as ``x``.

    Returns:
        Spearman's ``rho``, in ``[-1, 1]``.

    Raises:
        ValidationError: If the arrays differ in length or have fewer
            than 2 paired samples.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    _validate_pair(x, y)

    from scipy.stats import spearmanr

    rho, _ = spearmanr(x, y)
    return float(rho)


def correlation_summary(
    quantity_label: str,
    pairs: dict[str, tuple[str, np.ndarray, np.ndarray]],
) -> list[CorrelationResult]:
    """Build a sensitivity-summary table of one output quantity against several parameters.

    Args:
        quantity_label: The output quantity's display label (shared by
            every entry in this summary).
        pairs: ``{parameter_path: (parameter_label, x_samples, y_samples)}``
            for every parameter to include, ``x_samples``/``y_samples``
            paired by sample.

    Returns:
        One :class:`CorrelationResult` per entry, sorted by descending
        ``|pearson_r|`` -- the largest-magnitude linear association
        first. This ordering is a descriptive statistical ranking, not
        an engineering-importance ranking.
    """
    results = []
    for path, (label, x, y) in pairs.items():
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        pearson_r = pearson_correlation(x, y)
        try:
            spearman_rho = spearman_correlation(x, y)
        except ValidationError:
            spearman_rho = None
        results.append(
            CorrelationResult(
                parameter_path=path,
                parameter_label=label,
                quantity_label=quantity_label,
                pearson_r=pearson_r,
                spearman_rho=spearman_rho,
                n_samples=int(x.size),
            )
        )
    results.sort(key=lambda result: abs(result.pearson_r), reverse=True)
    return results


__all__ = [
    "CorrelationResult",
    "correlation_summary",
    "pearson_correlation",
    "spearman_correlation",
]
