"""Validation: TET4 3D patch test -- constant-strain reproduction.

The 3D analogue of tests/validation/test_cst_patch.py: a linear nodal
displacement field must be reproduced *exactly* by any tetrahedron,
regardless of shape or orientation, driven through the complete analysis
workflow (mesh -> boundary conditions -> StaticLinearAnalysis -> AnalysisResult).
"""

import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Mesh, Node, Tet4Element3D

YOUNGS_MODULUS = 70e9
POISSON_RATIO = 0.33

# A linear displacement field: u = a.x, v = b.x, w = c.x (x = [x,y,z]).
A = (0.004, -0.002, 0.0015)
B = (0.001, 0.003, -0.001)
C = (-0.0005, 0.0007, 0.002)
EXPECTED_STRAIN = [A[0], B[1], C[2], A[1] + B[0], B[2] + C[1], A[2] + C[0]]

TETRAHEDRON_GEOMETRIES = [
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),  # right tet
    ((1.0, 1.0, 1.0), (3.0, 1.5, 1.0), (1.5, 4.0, 2.0), (2.0, 2.0, 5.0)),  # arbitrary tet
    ((-1.0, -1.0, -1.0), (1.0, -1.0, -1.0), (0.0, 1.0, -1.0), (0.0, 0.0, 2.0)),  # skewed
]


def _build_result(geometry: tuple[tuple[float, float, float], ...]):
    material = LinearElastic3D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, density=2700.0
    )
    nodes = [Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(geometry)]
    element = Tet4Element3D(id=1, nodes=tuple(nodes), material=material)

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


@pytest.mark.parametrize("geometry", TETRAHEDRON_GEOMETRIES)
def test_patch_test_reproduces_exact_constant_strain(
    geometry: tuple[tuple[float, float, float], ...],
) -> None:
    result = _build_result(geometry)
    strain = result.element_strain(1)
    assert_allclose(strain, EXPECTED_STRAIN, atol=1e-12)


@pytest.mark.parametrize("geometry", TETRAHEDRON_GEOMETRIES)
def test_patch_test_stress_is_consistent_with_constitutive_law(
    geometry: tuple[tuple[float, float, float], ...],
) -> None:
    result = _build_result(geometry)
    material = LinearElastic3D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, density=2700.0
    )

    strain = result.element_strain(1)
    stress = result.element_stress(1)
    assert_allclose(stress, material.constitutive_matrix @ strain, atol=1e-3)


def test_patch_test_result_independent_of_tetrahedron_shape() -> None:
    strains = [_build_result(geometry).element_strain(1) for geometry in TETRAHEDRON_GEOMETRIES]
    for strain in strains:
        assert_allclose(strain, EXPECTED_STRAIN, atol=1e-12)
