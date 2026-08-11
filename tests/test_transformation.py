"""Tests for the 2D frame transformation matrix."""

import math

import numpy as np
from numpy.testing import assert_allclose

from femtoolkit.analysis.transformation import frame_transformation_matrix_2d


def test_transformation_matrix_shape() -> None:
    transformation = frame_transformation_matrix_2d(cos_theta=1.0, sin_theta=0.0)

    assert transformation.shape == (6, 6)


def test_transformation_matrix_is_orthogonal() -> None:
    """T^T * T must equal the identity matrix for any valid direction cosines."""
    transformation = frame_transformation_matrix_2d(cos_theta=0.6, sin_theta=0.8)

    assert_allclose(transformation.T @ transformation, np.eye(6), atol=1e-12)


def test_horizontal_transformation_is_identity() -> None:
    """c=1, s=0: local and global axes coincide."""
    transformation = frame_transformation_matrix_2d(cos_theta=1.0, sin_theta=0.0)

    assert_allclose(transformation, np.eye(6))


def test_vertical_transformation() -> None:
    """c=0, s=1: local axial axis (u) points along global Y."""
    transformation = frame_transformation_matrix_2d(cos_theta=0.0, sin_theta=1.0)

    expected = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert_allclose(transformation, expected)


def test_45_degree_transformation() -> None:
    c = s = math.sqrt(2) / 2
    transformation = frame_transformation_matrix_2d(cos_theta=c, sin_theta=s)

    assert_allclose(transformation[0, 0], c)
    assert_allclose(transformation[0, 1], s)
    assert_allclose(transformation[1, 0], -s)
    assert_allclose(transformation[1, 1], c)
    assert_allclose(transformation[2, 2], 1.0)
    assert_allclose(transformation[5, 5], 1.0)


def test_rotational_dofs_are_unaffected_by_orientation() -> None:
    """The theta/rz rows and columns are always the identity, in any orientation."""
    transformation = frame_transformation_matrix_2d(cos_theta=0.28, sin_theta=0.96)

    assert_allclose(transformation[2, :], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert_allclose(transformation[5, :], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert_allclose(transformation[:, 2], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert_allclose(transformation[:, 5], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
