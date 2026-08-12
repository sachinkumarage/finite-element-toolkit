"""Validation Case 5: two-triangle rectangular plate under uniaxial tension.

    Node 4 -------- Node 3
     |             / |
     |           /   |
     |         /     |
     |       /       |
     |     /         |
    Node 1 -------- Node 2

    Node 1 = (0, 0), Node 2 = (2, 0), Node 3 = (2, 1), Node 4 = (0, 1)
    Diagonal: Node 1 -- Node 3, splitting the rectangle into two CST
    elements: "lower" (1, 2, 3) and "upper" (1, 3, 4).

    E = 200 GPa, v = 0.3 (plane stress), t = 0.01 m
    Node 1: fixed (ux = uy = 0), Node 4: rollered (ux = 0)
    Fx = 500 N each at Node 2 and Node 3 (equivalent to a uniform 100 kPa
    tensile traction applied to the right edge, since Fx_total / (H*t) =
    1000 N / (1 m * 0.01 m) = 100000 Pa)

This case validates:

* **Global assembly** across two elements sharing an edge (Node 1, Node 3).
* **Displacement continuity** at the shared edge (implicit: both elements
  read the same global displacement vector at Node 1 and Node 3, since
  there is only one degree of freedom per node in the assembled system,
  not one per element).
* **Reaction equilibrium**: the fixed and rollered supports must resist
  exactly the applied edge load.
* **Element strain and stress**: for this particular loading (a uniform
  edge traction, applied as its statically equivalent nodal loads), the
  exact elasticity solution is itself constant stress -- so both CST
  elements should recover the *same* uniform uniaxial stress state,
  matching the applied traction, to high numerical precision.
"""

from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import CSTElement2D, Mesh, Node

YOUNGS_MODULUS = 200e9
POISSON_RATIO = 0.3
THICKNESS = 0.01
WIDTH = 2.0
HEIGHT = 1.0
APPLIED_LOAD = 1000.0  # N, total, split evenly between Node 2 and Node 3

EXPECTED_SIGMA_X = APPLIED_LOAD / (HEIGHT * THICKNESS)


def _build_result():
    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, formulation="plane_stress"
    )
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=WIDTH, y=0.0, z=0.0)
    node_3 = Node(id=3, x=WIDTH, y=HEIGHT, z=0.0)
    node_4 = Node(id=4, x=0.0, y=HEIGHT, z=0.0)

    lower = CSTElement2D(
        id=1, nodes=(node_1, node_2, node_3), material=material, thickness=THICKNESS
    )
    upper = CSTElement2D(
        id=2, nodes=(node_1, node_3, node_4), material=material, thickness=THICKNESS
    )

    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4):
        mesh.add_node(node)
    for element in (lower, upper):
        mesh.add_element(element)

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.X, value=0.0))
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.Y, value=0.0))
    analysis.add_boundary_condition(BoundaryCondition(node_id=4, dof=TranslationDOF.X, value=0.0))
    analysis.add_load(NodalLoad(node_id=2, dof=TranslationDOF.X, value=APPLIED_LOAD / 2))
    analysis.add_load(NodalLoad(node_id=3, dof=TranslationDOF.X, value=APPLIED_LOAD / 2))
    return analysis.solve()


def test_both_elements_recover_the_same_uniform_stress() -> None:
    result = _build_result()

    lower_stress = result.element_stress(1)
    upper_stress = result.element_stress(2)

    assert_allclose(lower_stress, upper_stress, atol=1e-3)


def test_stress_matches_the_applied_uniaxial_traction() -> None:
    result = _build_result()

    sigma_x, sigma_y, tau_xy = result.element_stress(1)
    assert_allclose(sigma_x, EXPECTED_SIGMA_X, rtol=1e-6)
    assert_allclose(sigma_y, 0.0, atol=1.0)
    assert_allclose(tau_xy, 0.0, atol=1.0)


def test_strain_is_consistent_between_elements() -> None:
    """Displacement continuity at the shared edge (Node 1, Node 3) implies
    both elements, having the same material, must report the same strain.
    """
    result = _build_result()

    assert_allclose(result.element_strain(1), result.element_strain(2), atol=1e-9)


def test_reactions_balance_the_applied_load() -> None:
    result = _build_result()

    rx1, ry1 = result.node_reaction(1)
    rx4, ry4 = result.node_reaction(4)

    assert_allclose(rx1 + rx4 + APPLIED_LOAD, 0.0, atol=1e-6)
    assert_allclose(ry1 + ry4, 0.0, atol=1e-6)


def test_roller_support_carries_no_vertical_reaction() -> None:
    """Node 4 is only restrained in X; its Y reaction must be zero."""
    result = _build_result()

    _, ry4 = result.node_reaction(4)
    assert_allclose(ry4, 0.0, atol=1e-6)


def test_von_mises_equals_the_uniaxial_stress() -> None:
    result = _build_result()

    assert_allclose(result.element_von_mises(1), EXPECTED_SIGMA_X, rtol=1e-6)
    assert_allclose(result.element_von_mises(2), EXPECTED_SIGMA_X, rtol=1e-6)
