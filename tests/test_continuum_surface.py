"""Tests for femtoolkit.continuum.surface (face tables and surface integration, Version 21)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.continuum.surface import (
    HEX8_FACES,
    TET4_FACES,
    quad_face_area,
    quad_face_consistent_matrix,
    quad_face_load_vector,
    quad_face_surface_jacobian,
    triangle_face_area,
    triangle_face_conductance_matrix,
    triangle_face_load_vector,
)
from femtoolkit.exceptions import DegenerateElementError

_UNIT_CUBE = np.array(
    [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ]
)


def test_tet4_faces_cover_every_node_and_are_distinct() -> None:
    assert len(TET4_FACES) == 4
    for face in TET4_FACES:
        assert len(set(face)) == 3
    assert {node for face in TET4_FACES for node in face} == {0, 1, 2, 3}


def test_hex8_faces_cover_every_node_twice() -> None:
    assert len(HEX8_FACES) == 6
    for face in HEX8_FACES:
        assert len(set(face)) == 4
    counts = np.bincount([node for face in HEX8_FACES for node in face])
    assert list(counts) == [3] * 8


def test_triangle_face_area_right_triangle() -> None:
    area = triangle_face_area(
        np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    )
    assert area == pytest.approx(0.5)


def test_triangle_face_area_independent_of_orientation_in_3d() -> None:
    """A triangle tilted arbitrarily in 3D space still has the correct area."""
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([1.0, 0.0, 1.0])
    p2 = np.array([0.0, 1.0, 1.0])
    area = triangle_face_area(p0, p1, p2)
    expected = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
    assert area == pytest.approx(expected)


def test_triangle_face_area_rejects_degenerate_triangle() -> None:
    with pytest.raises(DegenerateElementError):
        triangle_face_area(
            np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])
        )


def test_triangle_face_conductance_matrix_sum_equals_coefficient_times_area() -> None:
    coefficient = 15.0
    p0, p1, p2 = np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]), np.array([0.0, 3.0, 0.0])
    matrix = triangle_face_conductance_matrix(p0, p1, p2, coefficient)
    area = triangle_face_area(p0, p1, p2)
    assert matrix.shape == (3, 3)
    assert_allclose(matrix, matrix.T)
    assert matrix.sum() == pytest.approx(coefficient * area)


def test_triangle_face_load_vector_splits_evenly() -> None:
    coefficient = 30.0
    p0, p1, p2 = np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0])
    vector = triangle_face_load_vector(p0, p1, p2, coefficient)
    area = triangle_face_area(p0, p1, p2)
    assert_allclose(vector, np.full(3, coefficient * area / 3.0))
    assert vector.sum() == pytest.approx(coefficient * area)


def test_quad_face_surface_jacobian_matches_area_for_axis_aligned_face() -> None:
    face = _UNIT_CUBE[list(HEX8_FACES[0])]  # bottom face, unit square, z=0
    jacobian = quad_face_surface_jacobian(face, 0.0, 0.0)
    # A unit square parameterized over xi,eta in [-1,1]: dA/dxi/deta = area/4 at every point.
    assert jacobian == pytest.approx(0.25)


def test_quad_face_area_unit_square_face() -> None:
    for face_indices in HEX8_FACES:
        face = _UNIT_CUBE[list(face_indices)]
        assert quad_face_area(face) == pytest.approx(1.0)


def test_quad_face_area_invariant_under_rigid_rotation() -> None:
    """Rotating the whole face in 3D must not change its computed area."""
    face = _UNIT_CUBE[list(HEX8_FACES[0])]
    theta = 0.7
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated_face = face @ rotation.T
    assert quad_face_area(rotated_face) == pytest.approx(quad_face_area(face))


def test_quad_face_area_scales_with_a_larger_face() -> None:
    face = _UNIT_CUBE[list(HEX8_FACES[0])] * 2.0
    assert quad_face_area(face) == pytest.approx(4.0)


def test_quad_face_area_rejects_degenerate_face() -> None:
    degenerate = np.zeros((4, 3))
    with pytest.raises(DegenerateElementError):
        quad_face_area(degenerate)


def test_quad_face_consistent_matrix_sum_equals_coefficient_times_area() -> None:
    coefficient = 12.0
    face = _UNIT_CUBE[list(HEX8_FACES[0])]
    matrix = quad_face_consistent_matrix(face, coefficient)
    assert matrix.shape == (4, 4)
    assert_allclose(matrix, matrix.T)
    assert matrix.sum() == pytest.approx(coefficient * quad_face_area(face))


def test_quad_face_load_vector_sum_equals_coefficient_times_area() -> None:
    coefficient = 40.0
    face = _UNIT_CUBE[list(HEX8_FACES[0])]
    vector = quad_face_load_vector(face, coefficient)
    assert vector.shape == (4,)
    assert vector.sum() == pytest.approx(coefficient * quad_face_area(face))
    # A unit-square face under uniform loading splits evenly across its four corners.
    assert_allclose(vector, np.full(4, coefficient * quad_face_area(face) / 4.0))
