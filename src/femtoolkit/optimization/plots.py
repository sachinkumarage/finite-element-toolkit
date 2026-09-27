"""Optimization visualizations: history traces and the Pareto front (Version 32).

Follows the exact convention every prior visualization module in this
toolkit already established (:mod:`femtoolkit.postprocessing.visualization`,
Version 22; :mod:`femtoolkit.uncertainty.plots`, Version 31): every
function here builds and returns a
:class:`matplotlib.figure.Figure` and never calls ``plt.show()``.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from femtoolkit.exceptions import ValidationError
from femtoolkit.optimization.evaluation import DesignEvaluation, DesignStatus
from femtoolkit.optimization.history import OptimizationHistory
from femtoolkit.optimization.objectives import Objective
from femtoolkit.optimization.variables import DesignVariable
from femtoolkit.postprocessing.visualization import plot_line


def plot_objective_history(history: OptimizationHistory, objective: Objective) -> Figure:
    """Plot the best-feasible objective value after each evaluation.

    Args:
        history: The optimization history to plot.
        objective: The objective to track.

    Returns:
        A :class:`matplotlib.figure.Figure` with one line plot.

    Raises:
        ValidationError: If ``history`` has no evaluations.
    """
    if history.n_evaluations == 0:
        raise ValidationError("plot_objective_history requires at least one evaluation.")

    series = history.best_so_far_series(objective)
    x_values = np.arange(1, len(series) + 1)
    y_values = np.array([value if value is not None else np.nan for value in series])
    units_suffix = f" ({objective.units})" if objective.units else ""
    return plot_line(
        x_values,
        y_values,
        xlabel="Evaluation",
        ylabel=f"Best feasible {objective.name}{units_suffix}",
        title=f"Objective History: {objective.name} ({objective.direction.value})",
    )


def plot_constraint_violation_history(history: OptimizationHistory) -> Figure:
    """Plot each evaluation's total constraint violation across the optimization run.

    Args:
        history: The optimization history to plot.

    Returns:
        A :class:`matplotlib.figure.Figure` with one line plot. A
        violation of ``0.0`` means the design was feasible.

    Raises:
        ValidationError: If ``history`` has no evaluations.
    """
    if history.n_evaluations == 0:
        raise ValidationError(
            "plot_constraint_violation_history requires at least one evaluation."
        )

    x_values = np.arange(1, history.n_evaluations + 1)
    y_values = np.array([evaluation.total_violation for evaluation in history.evaluations])
    return plot_line(
        x_values,
        y_values,
        xlabel="Evaluation",
        ylabel="Total constraint violation",
        title="Constraint Violation History",
    )


def plot_design_variable_history(history: OptimizationHistory, variable: DesignVariable) -> Figure:
    """Plot how one design variable's value changed across every evaluation.

    Args:
        history: The optimization history to plot.
        variable: The design variable to trace.

    Returns:
        A :class:`matplotlib.figure.Figure` with one line plot.

    Raises:
        ValidationError: If ``history`` has no evaluations.
    """
    if history.n_evaluations == 0:
        raise ValidationError("plot_design_variable_history requires at least one evaluation.")

    x_values = np.arange(1, history.n_evaluations + 1)
    y_values = np.array(
        [float(evaluation.design_variables[variable.name]) for evaluation in history.evaluations]
    )
    units_suffix = f" ({variable.units})" if variable.units else ""
    return plot_line(
        x_values,
        y_values,
        xlabel="Evaluation",
        ylabel=f"{variable.name}{units_suffix}",
        title=f"Design Variable History: {variable.name}",
    )


def plot_pareto_front(
    evaluations: list[DesignEvaluation],
    objectives: list[Objective],
    pareto: list[DesignEvaluation],
    baseline: DesignEvaluation | None = None,
) -> Figure:
    """Plot the Pareto front for a two-objective optimization.

    Feasible, infeasible, Pareto, and baseline designs are labeled
    distinctly; no point is given a subjective rating.

    Args:
        evaluations: Every evaluated design.
        objectives: Exactly two objectives to plot (the first on the
            horizontal axis, the second on the vertical axis).
        pareto: The non-dominated subset (see
            :func:`~femtoolkit.optimization.pareto.pareto_front`).
        baseline: The baseline design's evaluation, if it should be
            marked separately.

    Returns:
        A :class:`matplotlib.figure.Figure` with one scatter plot.

    Raises:
        ValidationError: If ``objectives`` does not have exactly two entries.
    """
    if len(objectives) != 2:
        raise ValidationError(
            f"plot_pareto_front requires exactly two objectives, got {len(objectives)}."
        )
    x_objective, y_objective = objectives

    figure = Figure(figsize=(7.0, 5.5))
    axes = figure.add_subplot(111)

    pareto_ids = {evaluation.design_id for evaluation in pareto}
    feasible_non_pareto = [
        evaluation
        for evaluation in evaluations
        if evaluation.status is DesignStatus.FEASIBLE and evaluation.design_id not in pareto_ids
    ]
    infeasible = [
        evaluation for evaluation in evaluations if evaluation.status is not DesignStatus.FEASIBLE
    ]

    def _xy(points: list[DesignEvaluation]) -> tuple[np.ndarray, np.ndarray]:
        x = np.array([point.objective_values[x_objective.name] for point in points])
        y = np.array([point.objective_values[y_objective.name] for point in points])
        return x, y

    if infeasible:
        # Infeasible designs have no guaranteed objective values (a FAILED/INVALID
        # design has none at all); only plot those with both objectives available.
        plottable_infeasible = [
            e
            for e in infeasible
            if x_objective.name in e.objective_values and y_objective.name in e.objective_values
        ]
        if plottable_infeasible:
            x, y = _xy(plottable_infeasible)
            axes.scatter(x, y, c="lightgray", marker="x", label="Infeasible", zorder=2)
    if feasible_non_pareto:
        x, y = _xy(feasible_non_pareto)
        axes.scatter(x, y, c="#4C72B0", marker="o", label="Feasible", zorder=3)
    if pareto:
        x, y = _xy(pareto)
        axes.scatter(x, y, c="#C44E52", marker="o", label="Pareto (non-dominated)", zorder=4)
    baseline_has_x = baseline is not None and x_objective.name in baseline.objective_values
    baseline_has_y = baseline is not None and y_objective.name in baseline.objective_values
    if baseline_has_x and baseline_has_y:
        axes.scatter(
            [baseline.objective_values[x_objective.name]],
            [baseline.objective_values[y_objective.name]],
            c="black",
            marker="*",
            s=150,
            label="Baseline",
            zorder=5,
        )

    x_units = f" ({x_objective.units})" if x_objective.units else ""
    y_units = f" ({y_objective.units})" if y_objective.units else ""
    axes.set_xlabel(f"{x_objective.name}{x_units} ({x_objective.direction.value})")
    axes.set_ylabel(f"{y_objective.name}{y_units} ({y_objective.direction.value})")
    axes.set_title("Pareto Front")
    axes.legend()
    axes.grid(visible=True, alpha=0.3)
    figure.tight_layout()
    return figure


__all__ = [
    "plot_constraint_violation_history",
    "plot_design_variable_history",
    "plot_objective_history",
    "plot_pareto_front",
]
