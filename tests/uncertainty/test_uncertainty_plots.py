"""Tests for femtoolkit.uncertainty.plots."""

import numpy as np
import pytest
from matplotlib.figure import Figure

from femtoolkit.exceptions import ValidationError
from femtoolkit.uncertainty.distributions import DeterministicDistribution, NormalDistribution
from femtoolkit.uncertainty.parameters import UncertainParameter
from femtoolkit.uncertainty.plots import plot_input_distribution, plot_output_histogram


def test_plot_output_histogram_returns_figure() -> None:
    values = np.random.default_rng(0).normal(200.0, 10.0, size=200)
    figure = plot_output_histogram(values, "Maximum displacement", units="m")
    assert isinstance(figure, Figure)
    assert figure.axes[0].get_xlabel() == "Maximum displacement (m)"


def test_plot_output_histogram_rejects_empty_values() -> None:
    with pytest.raises(ValidationError):
        plot_output_histogram(np.array([]), "x")


def test_plot_input_distribution_for_normal_parameter() -> None:
    parameter = UncertainParameter(
        path="material.youngs_modulus",
        label="Young's Modulus",
        distribution=NormalDistribution(200e9, 5e9),
        units="Pa",
    )
    figure = plot_input_distribution(parameter)
    assert isinstance(figure, Figure)
    assert "Young's Modulus" in figure.axes[0].get_title()


def test_plot_input_distribution_for_deterministic_parameter() -> None:
    parameter = UncertainParameter(
        path="material.density", label="Density", distribution=DeterministicDistribution(7850.0)
    )
    figure = plot_input_distribution(parameter)
    assert isinstance(figure, Figure)
