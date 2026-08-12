"""Validation Case 1 / Patch Test: constant-strain reproduction.

The defining mathematical property of the constant strain triangle is
right there in its name: for *any* linear nodal displacement field

.. code-block:: text

    u(x, y) = a*x + b*y
    v(x, y) = c*x + d*y

a CST element must reproduce the *exact* constant strain field

.. code-block:: text

    epsilon_x = a
    epsilon_y = d
    gamma_xy  = b + c

regardless of the triangle's shape, size, or orientation -- because the
shape functions are themselves linear, so the interpolated field is
exactly the prescribed linear field, and its (constant) gradient is
exactly ``[a, d, b+c]``, not an approximation of it.

This test drives the property through the *complete* analysis workflow
(mesh -> boundary conditions -> StaticLinearAnalysis -> AnalysisResult),
not just the underlying B-matrix formula (see tests/test_strain.py for
that unit-level check) -- every one of the element's six DOFs is
prescribed as a boundary condition, so the solver has nothing left to
solve for, and the recovered strain must still match the formula exactly.
This is the classical **patch test**: it exercises geometry, DOF mapping,
assembly, boundary-condition application, and strain recovery together,
on three differently shaped triangles.
"""

import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import CSTElement2D, Mesh, Node

YOUNGS_MODULUS = 70e9
POISSON_RATIO = 0.33
THICKNESS = 0.005

# A linear displacement field: u = a*x + b*y, v = c*x + d*y.
A, B, C, D = 0.004, -0.002, 0.0015, 0.003
EXPECTED_STRAIN = [A, D, B + C]

TRIANGLE_GEOMETRIES = [
    ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),  # right triangle
    ((2.0, 1.0), (5.0, 2.0), (3.0, 6.0)),  # arbitrary triangle
    ((-1.0, -1.0), (1.0, -1.0), (0.0, 2.0)),  # isoceles, negative coordinates
]


def _build_result(geometry: tuple[tuple[float, float], ...]):
    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, formulation="plane_stress"
    )
    nodes = [Node(id=i + 1, x=x, y=y, z=0.0) for i, (x, y) in enumerate(geometry)]
    element = CSTElement2D(id=1, nodes=tuple(nodes), material=material, thickness=THICKNESS)

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


@pytest.mark.parametrize("geometry", TRIANGLE_GEOMETRIES)
def test_patch_test_reproduces_exact_constant_strain(
    geometry: tuple[tuple[float, float], ...],
) -> None:
    result = _build_result(geometry)

    strain = result.element_strain(1)
    assert_allclose(strain, EXPECTED_STRAIN, atol=1e-14)


@pytest.mark.parametrize("geometry", TRIANGLE_GEOMETRIES)
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


def test_patch_test_result_independent_of_triangle_shape() -> None:
    """The whole point of the patch test: the SAME prescribed field must
    give the SAME strain on every differently-shaped triangle.
    """
    strains = [_build_result(geometry).element_strain(1) for geometry in TRIANGLE_GEOMETRIES]

    for strain in strains:
        assert_allclose(strain, EXPECTED_STRAIN, atol=1e-14)
