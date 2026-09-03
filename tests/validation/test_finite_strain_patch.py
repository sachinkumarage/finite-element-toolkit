"""Validation: finite-deformation patch tests (spec section 25).

Extends the small-strain patch test tradition (tests/validation/test_tet4_patch.py,
test_hex8_patch.py) to finite deformation: a prescribed *homogeneous*
deformation gradient F (uniform everywhere, exactly reproducible by a
single linear/trilinear element regardless of shape) must give the exact
analytical Green-Lagrange strain and second Piola-Kirchhoff stress -- for
both TET4 and HEX8, driven through the geometrically nonlinear dispatch
functions directly (not the full Newton-Raphson solver, since a prescribed
displacement field applied to every DOF leaves nothing to iterate for --
matching test_tet4_patch.py's own approach).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis.geometric_nonlinear import (
    hex8_geometric_internal_force_and_tangent,
    initial_element_state,
    tet4_geometric_internal_force_and_tangent,
)
from femtoolkit.continuum.constitutive import isotropic_3d_matrix
from femtoolkit.continuum.deformation import green_lagrange_strain_voigt
from femtoolkit.materials import LinearElastic3D, SaintVenantKirchhoff3D
from femtoolkit.mesh import Hex8Element3D, Node, Tet4Element3D

YOUNGS_MODULUS = 70e9
POISSON_RATIO = 0.3

_TET4_COORDS = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
_HEX8_COORDS = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.0),
]

_DEFORMATIONS = {
    "uniaxial_extension": np.diag([1.3, 1.0, 1.0]),
    "triaxial_extension": np.diag([1.2, 1.1, 0.95]),
    "simple_shear": np.array([[1.0, 0.15, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
}


def _displacements_for(
    coords: list[tuple[float, float, float]], f_prescribed: np.ndarray
) -> np.ndarray:
    reference = np.array(coords)
    current = reference @ f_prescribed.T
    return (current - reference).flatten()


@pytest.mark.parametrize("deformation_name", list(_DEFORMATIONS))
def test_tet4_homogeneous_deformation_patch(deformation_name: str) -> None:
    f_prescribed = _DEFORMATIONS[deformation_name]
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_TET4_COORDS))
    placeholder = LinearElastic3D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, density=2700.0
    )
    tet = Tet4Element3D(id=1, nodes=nodes, material=placeholder)

    material = SaintVenantKirchhoff3D(youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO)
    committed = initial_element_state(tet, material)
    displacements = _displacements_for(_TET4_COORDS, f_prescribed)

    _, _, trial_state = tet4_geometric_internal_force_and_tangent(
        tet, material, displacements, committed
    )

    expected_strain = green_lagrange_strain_voigt(f_prescribed)
    expected_stress = isotropic_3d_matrix(YOUNGS_MODULUS, POISSON_RATIO) @ expected_strain

    assert_allclose(trial_state.states[0].strain, expected_strain, atol=1e-12)
    assert_allclose(trial_state.states[0].stress, expected_stress, rtol=1e-8)


@pytest.mark.parametrize("deformation_name", list(_DEFORMATIONS))
def test_hex8_homogeneous_deformation_patch(deformation_name: str) -> None:
    f_prescribed = _DEFORMATIONS[deformation_name]
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_HEX8_COORDS))
    placeholder = LinearElastic3D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, density=2700.0
    )
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)

    material = SaintVenantKirchhoff3D(youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO)
    committed = initial_element_state(hexa, material)
    displacements = _displacements_for(_HEX8_COORDS, f_prescribed)

    _, _, trial_state = hex8_geometric_internal_force_and_tangent(
        hexa, material, displacements, committed
    )

    expected_strain = green_lagrange_strain_voigt(f_prescribed)
    expected_stress = isotropic_3d_matrix(YOUNGS_MODULUS, POISSON_RATIO) @ expected_strain

    # A homogeneous deformation must be reproduced *exactly* (to floating-point
    # precision) at every one of the 8 Gauss points -- not just on average.
    for state in trial_state.states:
        assert_allclose(state.strain, expected_strain, atol=1e-11)
        assert_allclose(state.stress, expected_stress, rtol=1e-8, atol=1.0)
