"""Validation Case 4: single triangle stiffness matrix, independently derived.

Geometry: Node 1 = (0, 0), Node 2 = (1, 0), Node 3 = (0, 1). Material:
``E = 1``, ``v = 0.3`` (plane stress), ``t = 1`` -- unit values chosen so
the hand-derived numbers below are as easy as possible to check by hand,
not because they are physically realistic.

The expected 6x6 stiffness matrix is computed here from the raw formulas
(``b_i``, ``c_i``, ``B``, ``D``, ``Ke = t*A*B^T*D*B``) written out
directly with NumPy -- **not** by calling
:func:`femtoolkit.continuum.strain.triangle_strain_displacement_matrix`,
:func:`femtoolkit.continuum.constitutive.plane_stress_matrix`, or
:func:`femtoolkit.analysis.stiffness.cst_element_stiffness` -- so this is
a genuinely independent check of the implementation, not a
tautological one.
"""

import numpy as np
from numpy.testing import assert_allclose

from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import CSTElement2D, Node

YOUNGS_MODULUS = 1.0
POISSON_RATIO = 0.3
THICKNESS = 1.0


def _independently_derived_stiffness() -> np.ndarray:
    x1, y1, x2, y2, x3, y3 = 0.0, 0.0, 1.0, 0.0, 0.0, 1.0

    two_a = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    area = two_a / 2.0

    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1

    b_matrix = (1.0 / two_a) * np.array(
        [
            [b1, 0.0, b2, 0.0, b3, 0.0],
            [0.0, c1, 0.0, c2, 0.0, c3],
            [c1, b1, c2, b2, c3, b3],
        ]
    )

    factor = YOUNGS_MODULUS / (1.0 - POISSON_RATIO**2)
    d_matrix = factor * np.array(
        [
            [1.0, POISSON_RATIO, 0.0],
            [POISSON_RATIO, 1.0, 0.0],
            [0.0, 0.0, (1.0 - POISSON_RATIO) / 2.0],
        ]
    )

    return THICKNESS * area * b_matrix.T @ d_matrix @ b_matrix


# The matrix from _independently_derived_stiffness(), transcribed as
# literal numbers (11 significant figures) -- a second, static reference
# independent of even the formula above being evaluated correctly at test
# time.
EXPECTED_STIFFNESS = np.array(
    [
        [0.74175824176, 0.35714285714, -0.54945054945, -0.19230769231, -0.19230769231, -0.16483516484],  # noqa: E501
        [0.35714285714, 0.74175824176, -0.16483516484, -0.19230769231, -0.19230769231, -0.54945054945],  # noqa: E501
        [-0.54945054945, -0.16483516484, 0.54945054945, 0.0, 0.0, 0.16483516484],
        [-0.19230769231, -0.19230769231, 0.0, 0.19230769231, 0.19230769231, 0.0],
        [-0.19230769231, -0.19230769231, 0.0, 0.19230769231, 0.19230769231, 0.0],
        [-0.16483516484, -0.54945054945, 0.16483516484, 0.0, 0.0, 0.54945054945],
    ]
)
"""Transcribed literal values (11 significant figures), a second static reference."""


def test_independent_derivation_matches_transcribed_literal_values() -> None:
    """Sanity check on the independent derivation itself before using it below."""
    assert_allclose(_independently_derived_stiffness(), EXPECTED_STIFFNESS, atol=1e-9)


def test_cst_element_stiffness_matches_independent_derivation() -> None:
    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, formulation="plane_stress"
    )
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=0.0, y=1.0, z=0.0)
    element = CSTElement2D(
        id=1, nodes=(node_1, node_2, node_3), material=material, thickness=THICKNESS
    )

    assert_allclose(element.stiffness_matrix, EXPECTED_STIFFNESS, atol=1e-9)


def test_cst_element_stiffness_matches_independent_derivation_formula() -> None:
    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, formulation="plane_stress"
    )
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=0.0, y=1.0, z=0.0)
    element = CSTElement2D(
        id=1, nodes=(node_1, node_2, node_3), material=material, thickness=THICKNESS
    )

    assert_allclose(element.stiffness_matrix, _independently_derived_stiffness(), atol=1e-9)
