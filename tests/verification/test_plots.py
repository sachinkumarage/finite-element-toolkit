"""Tests for femtoolkit.verification.plots (Version 29)."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from femtoolkit.verification.convergence import (
    MeshConvergenceLevel,
    MeshConvergenceSample,
    run_mesh_convergence_study,
)
from femtoolkit.verification.plots import (
    plot_error_vs_mesh_size,
    plot_mesh_convergence,
    plot_numerical_vs_reference,
    plot_residual_history,
)


def _study():
    def level(label, size, value):
        return MeshConvergenceLevel(
            label=label, mesh_size=size, run=lambda: MeshConvergenceSample(10, 5, 20, value)
        )

    return run_mesh_convergence_study(
        "Tip displacement",
        [level("Coarse", 1.0, 1.0), level("Medium", 0.5, 1.1), level("Fine", 0.25, 1.11)],
    )


def test_plot_mesh_convergence_returns_figure() -> None:
    figure = plot_mesh_convergence(_study())
    assert isinstance(figure, Figure)
    assert len(figure.axes) == 1


def test_plot_error_vs_mesh_size_returns_figure() -> None:
    figure = plot_error_vs_mesh_size(_study())
    assert isinstance(figure, Figure)


def test_plot_error_vs_mesh_size_rejects_single_point_study() -> None:
    def level(label, size, value):
        return MeshConvergenceLevel(
            label=label, mesh_size=size, run=lambda: MeshConvergenceSample(10, 5, 20, value)
        )

    single_point_study = run_mesh_convergence_study("Q", [level("Only", 1.0, 1.0)])
    with pytest.raises(ValueError, match="at least two"):
        plot_error_vs_mesh_size(single_point_study)


def test_plot_residual_history_returns_figure() -> None:
    figure = plot_residual_history([1.0, 0.1, 0.01, 0.001], solver_name="Conjugate Gradient")
    assert isinstance(figure, Figure)


def test_plot_residual_history_rejects_empty_history() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        plot_residual_history([])


def test_plot_numerical_vs_reference_returns_figure() -> None:
    x_values = np.linspace(0.0, 1.0, 5)
    numerical = x_values * 1.01
    reference = x_values
    figure = plot_numerical_vs_reference(x_values, numerical, reference, "Displacement")
    assert isinstance(figure, Figure)
    assert figure.axes[0].get_legend() is not None
