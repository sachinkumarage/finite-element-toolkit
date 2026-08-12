"""Validation Cases 2 and 3 (plane stress half): constitutive matrix and uniaxial stress.

## Case 3: constitutive matrix

For a structural steel (``E = 200 GPa``, ``v = 0.3``), the plane-stress
constitutive matrix is hand-derived independently of the implementation
under test and compared directly.

## Case 2: uniaxial stress

A CST element under a prescribed displacement field consistent with a
known uniaxial stress state must recover that exact stress state. Rather
than solving a loaded, boundary-constrained model (which would introduce
Poisson-effect boundary artifacts unless very carefully set up), the
displacement field itself is derived analytically from the target stress:
for pure uniaxial stress ``sigma_x = S``, ``sigma_y = 0``, ``tau_xy = 0``,
isotropic plane-stress Hooke's law gives

.. code-block:: text

    epsilon_x = S / E
    epsilon_y = -v * S / E
    gamma_xy  = 0

so prescribing ``u = epsilon_x * x``, ``v = epsilon_y * y`` at every node
and recovering the resulting stress is a clean, boundary-artifact-free
check that ``B`` and ``D`` combine to reproduce the target stress state
exactly.
"""

import numpy as np
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, StaticLinearAnalysis, TranslationDOF
from femtoolkit.continuum.constitutive import plane_stress_matrix
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import CSTElement2D, Mesh, Node

YOUNGS_MODULUS = 200e9
POISSON_RATIO = 0.3


def test_plane_stress_constitutive_matrix_hand_derived() -> None:
    """D = E/(1-v^2) * [[1,v,0],[v,1,0],[0,0,(1-v)/2]], independently computed."""
    d_matrix = plane_stress_matrix(YOUNGS_MODULUS, POISSON_RATIO)

    factor = YOUNGS_MODULUS / (1.0 - POISSON_RATIO**2)
    expected = factor * np.array(
        [
            [1.0, POISSON_RATIO, 0.0],
            [POISSON_RATIO, 1.0, 0.0],
            [0.0, 0.0, (1.0 - POISSON_RATIO) / 2.0],
        ]
    )
    assert_allclose(d_matrix, expected, rtol=1e-12)


def _build_uniaxial_result(applied_stress: float):
    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, formulation="plane_stress"
    )
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=0.4, y=1.0, z=0.0)  # a non-right, non-isoceles triangle
    element = CSTElement2D(id=1, nodes=(node_1, node_2, node_3), material=material, thickness=0.01)

    mesh = Mesh()
    for node in (node_1, node_2, node_3):
        mesh.add_node(node)
    mesh.add_element(element)

    epsilon_x = applied_stress / YOUNGS_MODULUS
    epsilon_y = -POISSON_RATIO * epsilon_x

    analysis = StaticLinearAnalysis(mesh)
    for node in (node_1, node_2, node_3):
        u = epsilon_x * node.x
        v = epsilon_y * node.y
        analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.X, u))
        analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.Y, v))

    return analysis.solve()


def test_uniaxial_stress_is_recovered_exactly() -> None:
    applied_stress = 150e6  # 150 MPa
    result = _build_uniaxial_result(applied_stress)

    sigma_x, sigma_y, tau_xy = result.element_stress(1)
    assert_allclose(sigma_x, applied_stress, rtol=1e-9)
    assert_allclose(sigma_y, 0.0, atol=1e-3)
    assert_allclose(tau_xy, 0.0, atol=1e-3)


def test_uniaxial_stress_von_mises_equals_the_applied_stress() -> None:
    """Under pure uniaxial stress, von Mises equivalent stress equals the applied stress."""
    applied_stress = 150e6
    result = _build_uniaxial_result(applied_stress)

    assert_allclose(result.element_von_mises(1), applied_stress, rtol=1e-9)


def test_uniaxial_stress_principal_stresses() -> None:
    applied_stress = 150e6
    result = _build_uniaxial_result(applied_stress)

    sigma_1, sigma_2 = result.element_principal_stresses(1)
    assert_allclose(sigma_1, applied_stress, rtol=1e-9)
    assert_allclose(sigma_2, 0.0, atol=1e-3)
