"""Verification plots: mesh convergence, error, and solver residual history (Version 29).

Thin adapters over the existing Matplotlib-based plotting primitives in
:mod:`femtoolkit.postprocessing.visualization` (Version 22, extended
this version with :func:`~femtoolkit.postprocessing.visualization.plot_semilog_line`/
:func:`~femtoolkit.postprocessing.visualization.plot_comparison_line`)
-- this module contains no plotting logic of its own, only the
conversion from a verification data structure (a
:class:`~femtoolkit.verification.convergence.MeshConvergenceStudy`, a
residual history list, ...) into the ``x``/``y`` arrays those functions
expect.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from femtoolkit.postprocessing.visualization import (
    plot_comparison_line,
    plot_line,
    plot_semilog_line,
)
from femtoolkit.verification.convergence import MeshConvergenceStudy


def plot_mesh_convergence(study: MeshConvergenceStudy) -> Figure:
    """Plot a mesh convergence study's result quantity against mesh size.

    Args:
        study: The completed mesh convergence study to plot.

    Returns:
        A :class:`matplotlib.figure.Figure` with mesh size on the
        horizontal axis (finest mesh, i.e. smallest size, on the right)
        and the tracked result quantity on the vertical axis.
    """
    mesh_sizes = np.array([point.mesh_size for point in study.points])
    values = np.array([point.result_value for point in study.points])
    return plot_line(
        mesh_sizes,
        values,
        xlabel="Mesh size",
        ylabel=study.quantity,
        title=f"Mesh Convergence: {study.quantity}",
    )


def plot_error_vs_mesh_size(study: MeshConvergenceStudy) -> Figure:
    """Plot a mesh convergence study's relative change against mesh size.

    Args:
        study: The completed mesh convergence study to plot. Must have
            at least two points (the first point has no relative
            change to plot, since it has no predecessor).

    Returns:
        A :class:`matplotlib.figure.Figure` with mesh size on the
        horizontal axis and relative change between consecutive
        refinement levels on the (logarithmic) vertical axis.

    Raises:
        ValueError: If ``study`` has fewer than two points.
    """
    points_with_change = [point for point in study.points if point.relative_change is not None]
    if not points_with_change:
        raise ValueError("plot_error_vs_mesh_size requires at least two convergence points.")

    mesh_sizes = np.array([point.mesh_size for point in points_with_change])
    changes = np.array([point.relative_change for point in points_with_change])
    return plot_semilog_line(
        mesh_sizes,
        changes,
        xlabel="Mesh size",
        ylabel="Relative change",
        title=f"Convergence Error: {study.quantity}",
    )


def plot_residual_history(residual_history: list[float], solver_name: str = "Solver") -> Figure:
    """Plot an iterative solver's relative residual against iteration number.

    Args:
        residual_history: The per-iteration relative residual values,
            e.g. from
            ``solver_result.diagnostics["residual_history"]`` when a
            :class:`~femtoolkit.solvers.iterative.ConjugateGradientSolver`
            was constructed with ``track_residual_history=True``.
        solver_name: The solver's display name, for the plot title.

    Returns:
        A :class:`matplotlib.figure.Figure` with iteration number on the
        horizontal axis and relative residual on the (logarithmic)
        vertical axis.

    Raises:
        ValueError: If ``residual_history`` is empty.
    """
    if not residual_history:
        raise ValueError("plot_residual_history requires a non-empty residual history.")

    iterations = np.arange(1, len(residual_history) + 1)
    return plot_semilog_line(
        iterations,
        np.asarray(residual_history),
        xlabel="Iteration",
        ylabel="Relative residual",
        title=f"{solver_name}: Residual History",
    )


def plot_numerical_vs_reference(
    x_values: np.ndarray,
    numerical_values: np.ndarray,
    reference_values: np.ndarray,
    quantity: str,
    x_label: str = "Position",
) -> Figure:
    """Plot a numerical (FEA) series overlaid against a reference series.

    Args:
        x_values: The shared independent-variable values.
        numerical_values: The FEA (numerical) series.
        reference_values: The reference (analytical or validation
            dataset) series, same length as ``numerical_values``.
        quantity: The quantity being compared, for axis/title labels.
        x_label: Label for the horizontal axis.

    Returns:
        A :class:`matplotlib.figure.Figure` with both series plotted.
    """
    return plot_comparison_line(
        x_values,
        numerical_values,
        reference_values,
        xlabel=x_label,
        ylabel=quantity,
        title=f"Numerical vs. Reference: {quantity}",
        numerical_label="FEA result",
        reference_label="Reference",
    )


__all__ = [
    "plot_error_vs_mesh_size",
    "plot_mesh_convergence",
    "plot_numerical_vs_reference",
    "plot_residual_history",
]
