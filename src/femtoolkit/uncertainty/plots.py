"""Uncertainty visualizations: output histograms and input distributions (Version 31).

Follows the exact conventions
:mod:`femtoolkit.postprocessing.visualization` (Version 22) already
established: every function here builds and returns a
:class:`matplotlib.figure.Figure` and never calls ``plt.show()`` --
whether to display, embed, or save the figure is the caller's decision.
No new plotting dependency is introduced; Matplotlib has been a core
dependency since Version 22.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from femtoolkit.exceptions import ValidationError
from femtoolkit.uncertainty.distributions import DeterministicDistribution
from femtoolkit.uncertainty.parameters import UncertainParameter


def plot_output_histogram(
    values: np.ndarray, quantity_label: str, units: str = "", bins: int = 20
) -> Figure:
    """Plot a histogram of a Monte Carlo output quantity's successful sample values.

    Args:
        values: The successful runs' output values.
        quantity_label: A human-readable name for the quantity.
        units: A units string for the horizontal axis label.
        bins: The number of histogram bins.

    Returns:
        A :class:`matplotlib.figure.Figure` with one histogram.

    Raises:
        ValidationError: If ``values`` is empty.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValidationError("plot_output_histogram requires at least one value.")

    figure = Figure(figsize=(7.0, 5.0))
    axes = figure.add_subplot(111)
    axes.hist(values, bins=bins, color="#4C72B0", edgecolor="black", alpha=0.85)
    xlabel = f"{quantity_label} ({units})" if units else quantity_label
    axes.set_xlabel(xlabel)
    axes.set_ylabel("Frequency")
    axes.set_title(f"Output Distribution: {quantity_label}")
    axes.grid(visible=True, alpha=0.3)
    figure.tight_layout()
    return figure


def plot_input_distribution(parameter: UncertainParameter, n_points: int = 200) -> Figure:
    """Plot an uncertain parameter's probability distribution, before any simulation runs.

    A deterministic parameter (a point mass) has no density -- it is
    drawn as a single vertical marker at its value rather than a curve.
    Every other distribution is drawn as its probability density
    function over a range centered on its mean and wide enough to show
    its shape (bounded by the distribution's own hard bounds, if any).

    Args:
        parameter: The uncertain parameter to visualize.
        n_points: How many points to evaluate the density at.

    Returns:
        A :class:`matplotlib.figure.Figure`.
    """
    figure = Figure(figsize=(7.0, 5.0))
    axes = figure.add_subplot(111)
    distribution = parameter.distribution

    if isinstance(distribution, DeterministicDistribution):
        axes.axvline(distribution.value, color="#C44E52", linewidth=2.0)
        axes.set_ylim(0.0, 1.0)
        axes.set_yticks([])
        axes.annotate(
            f"{distribution.value:.6g}",
            xy=(distribution.value, 0.5),
            xytext=(10, 0),
            textcoords="offset points",
        )
    else:
        mean = distribution.mean()
        std = distribution.std()
        lower, upper = distribution.bounds()
        range_low = lower if lower is not None else mean - 4.0 * std
        range_high = upper if upper is not None else mean + 4.0 * std
        x = np.linspace(range_low, range_high, n_points)
        axes.plot(x, distribution.pdf(x), color="#4C72B0", linewidth=1.5)
        axes.fill_between(x, distribution.pdf(x), alpha=0.2, color="#4C72B0")

    units_suffix = f" ({parameter.units})" if parameter.units else ""
    axes.set_xlabel(f"{parameter.label}{units_suffix}")
    axes.set_ylabel("Probability Density")
    axes.set_title(f"Input Distribution: {parameter.label}")
    axes.grid(visible=True, alpha=0.3)
    figure.tight_layout()
    return figure


__all__ = ["plot_input_distribution", "plot_output_histogram"]
