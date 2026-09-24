"""Study visualizations: parameter vs. result quantity (Version 30).

Reuses :func:`~femtoolkit.postprocessing.visualization.plot_line`
directly -- the exact same 1D line-plot primitive the Version 22
post-processing module already provides -- rather than building a new
plotting function. A study plot is structurally the same thing as any
other 1D engineering plot: one series of x/y pairs, an x-label, a
y-label, and a title.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from femtoolkit.exceptions import ValidationError
from femtoolkit.postprocessing.visualization import plot_line
from femtoolkit.studies.extractors import Extractor
from femtoolkit.studies.parameter_sweep import ParameterDefinition
from femtoolkit.studies.results import StudyResult
from femtoolkit.studies.scenarios import get_by_path


def plot_study_quantity(
    result: StudyResult,
    parameter: ParameterDefinition,
    extractor: Extractor,
    quantity_label: str,
    title: str | None = None,
) -> Figure:
    """Plot one result quantity against one swept parameter, across every successful run.

    Args:
        result: The study's collected
            :class:`~femtoolkit.studies.results.StudyResult`.
        parameter: The swept parameter to plot on the horizontal axis;
            its value is read back from each run's immutable
            configuration snapshot (the exact value that run actually
            executed with), not from ``parameter.values``.
        extractor: The result-quantity extractor to plot on the
            vertical axis.
        quantity_label: A human-readable name for the plotted quantity
            (used as the vertical axis label).
        title: The plot title; defaults to ``"{quantity_label} vs.
            {parameter.label}"``.

    Returns:
        A :class:`matplotlib.figure.Figure` with one line plot.

    Raises:
        ValidationError: If fewer than one successful run is available,
            or the quantity is unavailable on any successful run.
    """
    runs = result.successful_runs
    if not runs:
        raise ValidationError("No successful run is available to plot.")

    parameter_values = [get_by_path(run.configuration_snapshot, parameter.path) for run in runs]
    quantity_values = []
    for run in runs:
        value = extractor(run)
        if value is None:
            raise ValidationError(
                f"Quantity {quantity_label!r} is not available on run {run.run_id!r}."
            )
        quantity_values.append(value)

    order = np.argsort(parameter_values)
    x_values = np.asarray(parameter_values)[order]
    y_values = np.asarray(quantity_values)[order]

    return plot_line(
        x_values,
        y_values,
        xlabel=parameter.label,
        ylabel=quantity_label,
        title=title or f"{quantity_label} vs. {parameter.label}",
    )


__all__ = ["plot_study_quantity"]
