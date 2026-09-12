"""Tests for edge equivalent nodal force computation (femtoolkit.continuum.edge)."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.continuum.edge import (
    edge_consistent_conductance_matrix,
    edge_equivalent_nodal_force,
    edge_shape_functions,
)
from femtoolkit.exceptions import DegenerateElementError, ValidationError


def test_edge_shape_functions_partition_of_unity() -> None:
    for xi in (-1.0, -0.3, 0.0, 0.5, 1.0):
        n1, n2 = edge_shape_functions(xi)
        assert_allclose(n1 + n2, 1.0)


def test_edge_shape_functions_nodal_property() -> None:
    n1_start, n2_start = edge_shape_functions(-1.0)
    assert_allclose(n1_start, 1.0)
    assert_allclose(n2_start, 0.0, atol=1e-12)

    n1_end, n2_end = edge_shape_functions(1.0)
    assert_allclose(n1_end, 0.0, atol=1e-12)
    assert_allclose(n2_end, 1.0)


def test_edge_shape_functions_midpoint() -> None:
    n1, n2 = edge_shape_functions(0.0)
    assert_allclose([n1, n2], [0.5, 0.5])


# --- Validation Case: length=1m, thickness=0.1m, traction=1000Pa -> F=100N ---


def test_edge_equivalent_nodal_force_matches_expected_total() -> None:
    force = edge_equivalent_nodal_force((0.0, 0.0), (1.0, 0.0), (0.0, 1000.0), 0.1)

    total_fy = force[1] + force[3]
    assert_allclose(total_fy, 100.0)


def test_edge_equivalent_nodal_force_splits_evenly_for_constant_traction() -> None:
    force = edge_equivalent_nodal_force((0.0, 0.0), (1.0, 0.0), (0.0, 1000.0), 0.1)

    fx1, fy1, fx2, fy2 = force
    assert_allclose(fy1, 50.0)
    assert_allclose(fy2, 50.0)
    assert_allclose(fx1, 0.0, atol=1e-12)
    assert_allclose(fx2, 0.0, atol=1e-12)


def test_edge_equivalent_nodal_force_x_traction() -> None:
    force = edge_equivalent_nodal_force((0.0, 0.0), (0.0, 2.0), (500.0, 0.0), 0.05)

    # L=2, thickness=0.05 -> F_total = 500*2*0.05 = 50 N, split 25/25.
    fx1, fy1, fx2, fy2 = force
    assert_allclose(fx1, 25.0)
    assert_allclose(fx2, 25.0)
    assert_allclose(fy1, 0.0, atol=1e-12)
    assert_allclose(fy2, 0.0, atol=1e-12)


def test_edge_equivalent_nodal_force_diagonal_edge() -> None:
    """A diagonal edge must use its true Euclidean length, not axis extents."""
    length = math.sqrt(2)
    thickness = 0.1
    traction = (1000.0, 0.0)

    force = edge_equivalent_nodal_force((0.0, 0.0), (1.0, 1.0), traction, thickness)

    expected_total = 1000.0 * length * thickness
    assert_allclose(force[0] + force[2], expected_total)
    assert_allclose(force[0], expected_total / 2.0)
    assert_allclose(force[2], expected_total / 2.0)


def test_edge_equivalent_nodal_force_scales_linearly_with_thickness() -> None:
    force_t1 = edge_equivalent_nodal_force((0.0, 0.0), (1.0, 0.0), (0.0, 1000.0), 0.1)
    force_t2 = edge_equivalent_nodal_force((0.0, 0.0), (1.0, 0.0), (0.0, 1000.0), 0.2)

    assert_allclose(force_t2, 2.0 * force_t1)


def test_edge_equivalent_nodal_force_scales_linearly_with_traction() -> None:
    force_1x = edge_equivalent_nodal_force((0.0, 0.0), (1.0, 0.0), (0.0, 1000.0), 0.1)
    force_2x = edge_equivalent_nodal_force((0.0, 0.0), (1.0, 0.0), (0.0, 2000.0), 0.1)

    assert_allclose(force_2x, 2.0 * force_1x)


def test_edge_equivalent_nodal_force_rejects_degenerate_edge() -> None:
    with pytest.raises(DegenerateElementError):
        edge_equivalent_nodal_force((0.0, 0.0), (0.0, 0.0), (0.0, 1000.0), 0.1)


@pytest.mark.parametrize("thickness", [0.0, -0.1, float("nan"), float("inf")])
def test_edge_equivalent_nodal_force_rejects_invalid_thickness(thickness: float) -> None:
    with pytest.raises(ValidationError):
        edge_equivalent_nodal_force((0.0, 0.0), (1.0, 0.0), (0.0, 1000.0), thickness)


def test_edge_equivalent_nodal_force_returns_length_4_array() -> None:
    force = edge_equivalent_nodal_force((0.0, 0.0), (1.0, 0.0), (0.0, 1000.0), 0.1)

    assert isinstance(force, np.ndarray)
    assert force.shape == (4,)


def test_edge_consistent_conductance_matrix_shape_and_symmetry() -> None:
    matrix = edge_consistent_conductance_matrix((0.0, 0.0), (1.0, 0.0), 25.0, 0.1)
    assert matrix.shape == (2, 2)
    assert_allclose(matrix, matrix.T)


def test_edge_consistent_conductance_matrix_row_sum_equals_coefficient_times_area() -> None:
    """integral(N^T N) summed over both rows/columns equals coefficient * area exactly."""
    coefficient, thickness, length = 25.0, 0.1, 2.0
    matrix = edge_consistent_conductance_matrix((0.0, 0.0), (length, 0.0), coefficient, thickness)
    assert_allclose(matrix.sum(), coefficient * thickness * length)


def test_edge_consistent_conductance_matrix_diagonal_dominant_pattern() -> None:
    """The standard consistent [[2,1],[1,2]] pattern, scaled by coefficient*thickness*L/6."""
    coefficient, thickness, length = 10.0, 0.5, 3.0
    matrix = edge_consistent_conductance_matrix((0.0, 0.0), (length, 0.0), coefficient, thickness)
    factor = coefficient * thickness * length / 6.0
    assert_allclose(matrix, factor * np.array([[2.0, 1.0], [1.0, 2.0]]))


def test_edge_consistent_conductance_matrix_scales_linearly_with_coefficient() -> None:
    m1 = edge_consistent_conductance_matrix((0.0, 0.0), (1.0, 0.0), 10.0, 0.1)
    m2 = edge_consistent_conductance_matrix((0.0, 0.0), (1.0, 0.0), 20.0, 0.1)
    assert_allclose(m2, 2.0 * m1)


def test_edge_consistent_conductance_matrix_rejects_degenerate_edge() -> None:
    with pytest.raises(DegenerateElementError):
        edge_consistent_conductance_matrix((0.0, 0.0), (0.0, 0.0), 10.0, 0.1)


@pytest.mark.parametrize("coefficient", [0.0, -1.0, float("nan"), float("inf")])
def test_edge_consistent_conductance_matrix_rejects_invalid_coefficient(coefficient: float) -> None:
    with pytest.raises(ValidationError):
        edge_consistent_conductance_matrix((0.0, 0.0), (1.0, 0.0), coefficient, 0.1)


@pytest.mark.parametrize("thickness", [0.0, -0.1, float("nan"), float("inf")])
def test_edge_consistent_conductance_matrix_rejects_invalid_thickness(thickness: float) -> None:
    with pytest.raises(ValidationError):
        edge_consistent_conductance_matrix((0.0, 0.0), (1.0, 0.0), 10.0, thickness)
