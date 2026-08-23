"""Tests for femtoolkit.continuum.mass: element consistent/lumped mass math."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.continuum.mass import (
    lumped_mass_matrix,
    quad_consistent_mass_matrix,
    quad_shape_function_matrix,
    triangle_consistent_mass_matrix,
)
from femtoolkit.exceptions import ValidationError

DENSITY = 1000.0
THICKNESS = 0.01


# --- CST consistent mass ---


def test_triangle_mass_has_correct_shape() -> None:
    m = triangle_consistent_mass_matrix(density=DENSITY, area=0.5, thickness=THICKNESS)
    assert m.shape == (6, 6)


def test_triangle_mass_is_symmetric() -> None:
    m = triangle_consistent_mass_matrix(density=DENSITY, area=0.5, thickness=THICKNESS)
    assert_allclose(m, m.T)


def test_triangle_mass_diagonal_is_positive() -> None:
    m = triangle_consistent_mass_matrix(density=DENSITY, area=0.5, thickness=THICKNESS)
    assert (np.diag(m) > 0).all()


def test_triangle_mass_total_matches_physical_mass() -> None:
    area = 0.5
    m = triangle_consistent_mass_matrix(density=DENSITY, area=area, thickness=THICKNESS)

    expected_total = DENSITY * area * THICKNESS
    x_indices = [0, 2, 4]
    y_indices = [1, 3, 5]
    assert_allclose(m[np.ix_(x_indices, x_indices)].sum(), expected_total)
    assert_allclose(m[np.ix_(y_indices, y_indices)].sum(), expected_total)


def test_triangle_mass_has_no_x_y_coupling() -> None:
    m = triangle_consistent_mass_matrix(density=DENSITY, area=0.5, thickness=THICKNESS)
    x_indices = [0, 2, 4]
    y_indices = [1, 3, 5]
    assert_allclose(m[np.ix_(x_indices, y_indices)], np.zeros((3, 3)))


def test_triangle_mass_scales_linearly_with_density_and_area() -> None:
    base = triangle_consistent_mass_matrix(density=1.0, area=1.0, thickness=1.0)
    scaled = triangle_consistent_mass_matrix(density=2.0, area=3.0, thickness=1.0)
    assert_allclose(scaled, 6.0 * base)


@pytest.mark.parametrize(
    "kwargs", [{"density": 0.0}, {"density": -1.0}, {"area": 0.0}, {"thickness": -0.01}]
)
def test_triangle_mass_rejects_non_positive_inputs(kwargs) -> None:
    defaults = {"density": DENSITY, "area": 0.5, "thickness": THICKNESS}
    defaults.update(kwargs)
    with pytest.raises(ValidationError):
        triangle_consistent_mass_matrix(**defaults)


# --- Q4 consistent mass ---


@pytest.fixture
def unit_square_coords():
    return [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]


def test_quad_mass_has_correct_shape(unit_square_coords) -> None:
    x, y = unit_square_coords
    m = quad_consistent_mass_matrix(x, y, density=DENSITY, thickness=THICKNESS)
    assert m.shape == (8, 8)


def test_quad_mass_is_symmetric(unit_square_coords) -> None:
    x, y = unit_square_coords
    m = quad_consistent_mass_matrix(x, y, density=DENSITY, thickness=THICKNESS)
    assert_allclose(m, m.T)


def test_quad_mass_diagonal_is_positive(unit_square_coords) -> None:
    x, y = unit_square_coords
    m = quad_consistent_mass_matrix(x, y, density=DENSITY, thickness=THICKNESS)
    assert (np.diag(m) > 0).all()


def test_quad_mass_total_matches_physical_mass_for_unit_square(unit_square_coords) -> None:
    x, y = unit_square_coords
    m = quad_consistent_mass_matrix(x, y, density=DENSITY, thickness=THICKNESS)

    expected_total = DENSITY * 1.0 * THICKNESS  # unit square area = 1
    x_indices = [0, 2, 4, 6]
    y_indices = [1, 3, 5, 7]
    assert_allclose(m[np.ix_(x_indices, x_indices)].sum(), expected_total)
    assert_allclose(m[np.ix_(y_indices, y_indices)].sum(), expected_total)


def test_quad_mass_has_no_x_y_coupling(unit_square_coords) -> None:
    x, y = unit_square_coords
    m = quad_consistent_mass_matrix(x, y, density=DENSITY, thickness=THICKNESS)
    x_indices = [0, 2, 4, 6]
    y_indices = [1, 3, 5, 7]
    assert_allclose(m[np.ix_(x_indices, y_indices)], np.zeros((4, 4)), atol=1e-12)


def test_quad_shape_function_matrix_shape() -> None:
    n = quad_shape_function_matrix((0.25, 0.25, 0.25, 0.25))
    assert n.shape == (2, 8)
    assert_allclose(n[0], [0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0])
    assert_allclose(n[1], [0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25])


def test_quad_mass_rejects_non_positive_density(unit_square_coords) -> None:
    x, y = unit_square_coords
    with pytest.raises(ValidationError):
        quad_consistent_mass_matrix(x, y, density=-1.0, thickness=THICKNESS)


# --- Lumped mass ---


def test_lumped_mass_is_diagonal() -> None:
    consistent = triangle_consistent_mass_matrix(density=DENSITY, area=0.5, thickness=THICKNESS)
    lumped = lumped_mass_matrix(consistent)

    off_diagonal = lumped - np.diag(np.diag(lumped))
    assert_allclose(off_diagonal, np.zeros_like(lumped))


def test_lumped_mass_conserves_total_triangle_mass() -> None:
    area = 0.5
    consistent = triangle_consistent_mass_matrix(density=DENSITY, area=area, thickness=THICKNESS)
    lumped = lumped_mass_matrix(consistent)

    expected_total = DENSITY * area * THICKNESS
    x_indices = [0, 2, 4]
    y_indices = [1, 3, 5]
    assert_allclose(np.diag(lumped)[x_indices].sum(), expected_total)
    assert_allclose(np.diag(lumped)[y_indices].sum(), expected_total)


def test_lumped_mass_conserves_total_quad_mass(unit_square_coords) -> None:
    x, y = unit_square_coords
    consistent = quad_consistent_mass_matrix(x, y, density=DENSITY, thickness=THICKNESS)
    lumped = lumped_mass_matrix(consistent)

    expected_total = DENSITY * 1.0 * THICKNESS
    x_indices = [0, 2, 4, 6]
    y_indices = [1, 3, 5, 7]
    assert_allclose(np.diag(lumped)[x_indices].sum(), expected_total)
    assert_allclose(np.diag(lumped)[y_indices].sum(), expected_total)


def test_lumped_mass_is_positive() -> None:
    consistent = triangle_consistent_mass_matrix(density=DENSITY, area=0.5, thickness=THICKNESS)
    lumped = lumped_mass_matrix(consistent)
    assert (np.diag(lumped) > 0).all()
