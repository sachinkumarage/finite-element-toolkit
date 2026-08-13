"""Validation: Q4 patch test -- constant-strain reproduction.

A linear displacement field, ``u = a*x + b*y``, ``v = c*x + d*y``, is a
special case of the Q4 element's bilinear interpolation (the bilinear
``xi*eta`` cross term simply has a zero coefficient), so a Q4 element
must reproduce the *exact* constant strain field

.. code-block:: text

    epsilon_x = a
    epsilon_y = d
    gamma_xy  = b + c

at every point within the element -- not just its center -- for any
quadrilateral shape, size, or orientation. This is the classical patch
test, driven through the complete analysis workflow (mesh -> boundary
conditions -> StaticLinearAnalysis -> AnalysisResult) with every one of
the element's eight DOFs prescribed directly, so the solver has nothing
left to solve for and the recovered (center) strain must still match the
formula exactly.
"""

import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, Node, QuadElement2D

YOUNGS_MODULUS = 70e9
POISSON_RATIO = 0.33
THICKNESS = 0.005

# A linear displacement field: u = a*x + b*y, v = c*x + d*y.
A, B, C, D = 0.004, -0.002, 0.0015, 0.003
EXPECTED_STRAIN = [A, D, B + C]

QUAD_GEOMETRIES = [
    ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),  # unit square
    ((0.0, 0.0), (2.0, 0.0), (2.5, 1.5), (0.3, 1.2)),  # general convex quad
    ((-1.0, -1.0), (1.0, -0.8), (1.2, 1.0), (-0.9, 1.1)),  # negative coordinates
]


def _build_result(geometry: tuple[tuple[float, float], ...]):
    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, formulation="plane_stress"
    )
    nodes = [Node(id=i + 1, x=x, y=y, z=0.0) for i, (x, y) in enumerate(geometry)]
    element = QuadElement2D(id=1, nodes=tuple(nodes), material=material, thickness=THICKNESS)

    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(element)

    analysis = StaticLinearAnalysis(mesh)
    for node in nodes:
        u = A * node.x + B * node.y
        v = C * node.x + D * node.y
        analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.X, u))
        analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.Y, v))

    return analysis.solve()


@pytest.mark.parametrize("geometry", QUAD_GEOMETRIES)
def test_patch_test_reproduces_exact_constant_strain(
    geometry: tuple[tuple[float, float], ...],
) -> None:
    result = _build_result(geometry)

    strain = result.element_strain(1)
    assert_allclose(strain, EXPECTED_STRAIN, atol=1e-10)


@pytest.mark.parametrize("geometry", QUAD_GEOMETRIES)
def test_patch_test_stress_is_consistent_with_constitutive_law(
    geometry: tuple[tuple[float, float], ...],
) -> None:
    result = _build_result(geometry)
    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, formulation="plane_stress"
    )

    strain = result.element_strain(1)
    stress = result.element_stress(1)
    assert_allclose(stress, material.constitutive_matrix @ strain, atol=1e-6)


def test_patch_test_result_independent_of_quad_shape() -> None:
    """The whole point of the patch test: the SAME prescribed field must
    give the SAME strain on every differently-shaped quadrilateral.
    """
    strains = [_build_result(geometry).element_strain(1) for geometry in QUAD_GEOMETRIES]

    for strain in strains:
        assert_allclose(strain, EXPECTED_STRAIN, atol=1e-10)
