"""Tests for femtoolkit.optimization.benchmarks.

The benchmark functions themselves (sphere/rosenbrock/rastrigin) are
pure NumPy -- these tests check them directly, with zero FEA
involvement, plus one integration test confirming the problem-builder
glue correctly wires a benchmark into a real, solvable
OptimizationProblem.
"""

from __future__ import annotations

import numpy as np
import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.optimization.algorithms import OptimizationConfig
from femtoolkit.optimization.benchmarks import (
    build_benchmark_problem,
    rastrigin,
    rosenbrock,
    sphere,
)
from femtoolkit.optimization.runner import OptimizationRunner


def test_sphere_minimum_at_origin() -> None:
    assert sphere(np.zeros(5)) == pytest.approx(0.0)


def test_sphere_positive_away_from_origin() -> None:
    assert sphere(np.array([1.0, -2.0, 3.0])) == pytest.approx(1.0 + 4.0 + 9.0)


def test_rosenbrock_minimum_at_ones() -> None:
    assert rosenbrock(np.ones(4)) == pytest.approx(0.0)


def test_rosenbrock_requires_two_dimensions() -> None:
    with pytest.raises(ValidationError):
        rosenbrock(np.array([1.0]))


def test_rosenbrock_known_value() -> None:
    # f(0, 0) = 100 * (0 - 0)^2 + (1 - 0)^2 = 1
    assert rosenbrock(np.array([0.0, 0.0])) == pytest.approx(1.0)


def test_rastrigin_minimum_at_origin() -> None:
    assert rastrigin(np.zeros(3)) == pytest.approx(0.0)


def test_rastrigin_known_value() -> None:
    # f(1, 0) = 10*2 + (1 - 10*cos(2*pi)) + (0 - 10*cos(0)) = 20 + (1 - 10) + (0 - 10) = 1
    assert rastrigin(np.array([1.0, 0.0])) == pytest.approx(1.0)


def test_build_benchmark_problem_rejects_zero_dimensions() -> None:
    with pytest.raises(ValidationError):
        build_benchmark_problem("bad", sphere, n_dimensions=0, lower_bound=-1.0, upper_bound=1.0)


def test_build_benchmark_problem_runs_end_to_end() -> None:
    problem = build_benchmark_problem(
        "sphere-smoke", sphere, n_dimensions=2, lower_bound=-2.0, upper_bound=2.0
    )
    config = OptimizationConfig(algorithm="coordinate_search", max_evaluations=20, seed=1)
    result = OptimizationRunner().run(problem, config)
    best = result.best_feasible()
    assert best is not None
    assert best.objective_values["f"] >= 0.0
