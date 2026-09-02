"""Validation: HEX8 3D patch test -- constant-strain reproduction.

The 3D analogue of tests/validation/test_quad_patch.py: a linear nodal
displacement field must be reproduced *exactly* even though HEX8's shape
functions are trilinear (not linear) and its B matrix varies pointwise --
the interpolated *field* of a purely linear boundary condition is still
exactly linear everywhere inside the element, so its gradient (strain) is
still exactly constant.
"""

from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node

YOUNGS_MODULUS = 70e9
POISSON_RATIO = 0.33

A = (0.004, -0.002, 0.0015)
B = (0.001, 0.003, -0.001)
C = (-0.0005, 0.0007, 0.002)
EXPECTED_STRAIN = [A[0], B[1], C[2], A[1] + B[0], B[2] + C[1], A[2] + C[0]]

_CUBE_COORDS = [
    (0.0, 0.0, 0.0),
    (2.0, 0.0, 0.0),
    (2.0, 1.5, 0.0),
    (0.0, 1.5, 0.0),
    (0.0, 0.0, 1.0),
    (2.0, 0.0, 1.0),
    (2.0, 1.5, 1.0),
    (0.0, 1.5, 1.0),
]


def _build_result():
    material = LinearElastic3D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, density=2700.0
    )
    nodes = [Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_CUBE_COORDS)]
    element = Hex8Element3D(id=1, nodes=tuple(nodes), material=material)

    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(element)

    analysis = StaticLinearAnalysis(mesh)
    for node in nodes:
        u = A[0] * node.x + A[1] * node.y + A[2] * node.z
        v = B[0] * node.x + B[1] * node.y + B[2] * node.z
        w = C[0] * node.x + C[1] * node.y + C[2] * node.z
        analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.X, u))
        analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.Y, v))
        analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.Z, w))

    return analysis.solve()


def test_patch_test_reproduces_exact_constant_strain() -> None:
    result = _build_result()
    strain = result.element_strain(1)
    assert_allclose(strain, EXPECTED_STRAIN, atol=1e-10)


def test_patch_test_stress_is_consistent_with_constitutive_law() -> None:
    result = _build_result()
    material = LinearElastic3D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, density=2700.0
    )

    strain = result.element_strain(1)
    stress = result.element_stress(1)
    assert_allclose(stress, material.constitutive_matrix @ strain, atol=1e-3)
