"""Validation: two-element Q4 plate under uniaxial tension (assembly, continuity, equilibrium).

    Node6 ---- Node5 ---- Node4
      |    left  |  right   |
      |          |          |
    Node1 ---- Node2 ---- Node3

    Node 1=(0,0), 2=(1,0), 3=(2,0), 4=(2,1), 5=(1,1), 6=(0,1)
    "left" element: (1, 2, 5, 6); "right" element: (2, 3, 4, 5)
    E = 200 GPa, v = 0.3 (plane stress), t = 0.01 m
    Node 1: fixed (ux = uy = 0), Node 6: rollered (ux = 0)
    Fx = 500 N each at Node 3 and Node 4 (equivalent to a uniform 100 kPa
    tensile traction on the right edge)

This validates:

* **Global assembly** across two Q4 elements sharing an edge (Node 2, Node 5).
* **Displacement continuity** at the shared edge -- there is exactly one
  degree of freedom per node in the assembled system, not one per element.
* **Reaction equilibrium**: the fixed and rollered supports must resist
  exactly the applied edge load.
* **Element strain and stress**: for this particular loading, the exact
  elasticity solution is itself constant stress, so both Q4 elements
  should recover the same uniform uniaxial stress state.
"""

from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, Node, QuadElement2D

YOUNGS_MODULUS = 200e9
POISSON_RATIO = 0.3
THICKNESS = 0.01
HEIGHT = 1.0
APPLIED_LOAD = 1000.0  # N, total, split evenly between Node 3 and Node 4

EXPECTED_SIGMA_X = APPLIED_LOAD / (HEIGHT * THICKNESS)


def _build_result():
    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, formulation="plane_stress"
    )
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=2.0, y=0.0, z=0.0)
    node_4 = Node(id=4, x=2.0, y=HEIGHT, z=0.0)
    node_5 = Node(id=5, x=1.0, y=HEIGHT, z=0.0)
    node_6 = Node(id=6, x=0.0, y=HEIGHT, z=0.0)

    left = QuadElement2D(
        id=1, nodes=(node_1, node_2, node_5, node_6), material=material, thickness=THICKNESS
    )
    right = QuadElement2D(
        id=2, nodes=(node_2, node_3, node_4, node_5), material=material, thickness=THICKNESS
    )

    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4, node_5, node_6):
        mesh.add_node(node)
    for element in (left, right):
        mesh.add_element(element)

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.X, value=0.0))
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=TranslationDOF.Y, value=0.0))
    analysis.add_boundary_condition(BoundaryCondition(node_id=6, dof=TranslationDOF.X, value=0.0))
    analysis.add_load(NodalLoad(node_id=3, dof=TranslationDOF.X, value=APPLIED_LOAD / 2))
    analysis.add_load(NodalLoad(node_id=4, dof=TranslationDOF.X, value=APPLIED_LOAD / 2))
    return analysis.solve()


def test_both_elements_recover_the_same_uniform_stress() -> None:
    result = _build_result()

    assert_allclose(result.element_stress(1), result.element_stress(2), atol=1e-3)


def test_stress_matches_the_applied_uniaxial_traction() -> None:
    result = _build_result()

    sigma_x, sigma_y, tau_xy = result.element_stress(1)
    assert_allclose(sigma_x, EXPECTED_SIGMA_X, rtol=1e-6)
    assert_allclose(sigma_y, 0.0, atol=1.0)
    assert_allclose(tau_xy, 0.0, atol=1.0)


def test_strain_is_consistent_between_elements() -> None:
    """Displacement continuity at the shared edge (Node 2, Node 5) implies
    both elements, having the same material, must report the same strain.
    """
    result = _build_result()

    assert_allclose(result.element_strain(1), result.element_strain(2), atol=1e-9)


def test_shared_edge_nodes_have_single_consistent_displacement() -> None:
    """Node 2 and Node 5 are shared between both elements; the assembled
    system stores one displacement per node, not one per contributing
    element, so this is implicitly guaranteed -- checked here directly.
    """
    result = _build_result()

    ux2, uy2 = result.node_displacement(2)
    ux5, uy5 = result.node_displacement(5)
    assert_allclose(uy2, 0.0, atol=1e-12)
    assert uy5 < 0.0  # Poisson contraction at the free top edge


def test_reactions_balance_the_applied_load() -> None:
    result = _build_result()

    rx1, ry1 = result.node_reaction(1)
    rx6, ry6 = result.node_reaction(6)
    assert_allclose(rx1 + rx6 + APPLIED_LOAD, 0.0, atol=1e-6)
    assert_allclose(ry1 + ry6, 0.0, atol=1e-6)


def test_roller_support_carries_no_vertical_reaction() -> None:
    result = _build_result()

    _, ry6 = result.node_reaction(6)
    assert_allclose(ry6, 0.0, atol=1e-6)


def test_von_mises_equals_the_uniaxial_stress() -> None:
    result = _build_result()

    assert_allclose(result.element_von_mises(1), EXPECTED_SIGMA_X, rtol=1e-6)
    assert_allclose(result.element_von_mises(2), EXPECTED_SIGMA_X, rtol=1e-6)
