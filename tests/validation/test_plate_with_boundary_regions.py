"""Validation Case 3: rectangular plate, fixed left boundary, right-side traction.

    Node ... (left, fixed: ux=uy=0) --------- (right, traction) ... Node

    Rectangle: width = 2 m, height = 1 m, Q4 mesh (nx=4, ny=2)
    E = 200 GPa, v = 0.3 (plane stress), t = 0.01 m
    Left boundary: fixed (ux = uy = 0 at every node)
    Right boundary: normal traction = 500 kPa (tension, pulling outward)

This exercises the complete Version 9 workflow -- Rectangle geometry,
named boundaries, automatic node/edge selection, boundary-region boundary
conditions, and a distributed traction converted to equivalent nodal
loads -- solved through the unmodified Version 3 solver.

Analytical comparison (uniaxial tension of a plate, E, cross-section
height*thickness):

.. code-block:: text

    sigma_x = traction                              (exact: pure force equilibrium)
    ux(right) ~ traction * width / E                (approximate: see note below)

**Why the displacement comparison is approximate, not exact.** Fixing
*every* left-edge node independently (``ux=uy=0`` at each) over-restrains
the plate's Poisson contraction right at the support, creating a local
boundary disturbance (a discrete analogue of Saint-Venant's principle)
that the idealized 1D bar formula does not model. sigma_x, by contrast,
is exact everywhere because it follows directly from global equilibrium
in the X direction, which the support detail cannot change.
"""

from numpy.testing import assert_allclose

from femtoolkit.analysis import DistributedLoad, LoadCase
from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import create_quad_mesh

YOUNGS_MODULUS = 200e9
POISSON_RATIO = 0.3
THICKNESS = 0.01
WIDTH = 2.0
HEIGHT = 1.0
TRACTION = 500e3  # 500 kPa


def _build_result():
    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, formulation="plane_stress"
    )
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=4, ny=2, material=material, thickness=THICKNESS
    )
    domain = Rectangle(width=WIDTH, height=HEIGHT)

    load_case = LoadCase(name="Tension", mesh=mesh)
    load_case.fix_boundary(domain.boundary("left"), ux=0.0, uy=0.0)
    load_case.add_distributed_load(DistributedLoad(domain.boundary("right"), magnitude=TRACTION))

    return load_case.solve(), mesh, domain


def test_every_element_recovers_the_exact_applied_traction() -> None:
    result, mesh, _ = _build_result()

    for element in mesh.elements:
        sigma_x, _, _ = result.element_stress(element.id)
        assert_allclose(sigma_x, TRACTION, rtol=1e-9)


def test_right_edge_displacement_approximately_matches_uniaxial_theory() -> None:
    result, mesh, domain = _build_result()

    right_nodes = mesh.nodes_on_boundary(domain.boundary("right"))
    average_ux = sum(result.node_displacement(n.id)[0] for n in right_nodes) / len(right_nodes)

    analytical_ux = TRACTION * WIDTH / YOUNGS_MODULUS
    # Within 2%: the discrete fully-fixed left edge introduces a local
    # boundary disturbance the 1D bar formula does not capture (see
    # module docstring), but the two must still be close.
    assert_allclose(average_ux, analytical_ux, rtol=0.02)


def test_left_boundary_nodes_have_zero_displacement() -> None:
    result, mesh, domain = _build_result()

    for node in mesh.nodes_on_boundary(domain.boundary("left")):
        ux, uy = result.node_displacement(node.id)
        assert_allclose(ux, 0.0, atol=1e-12)
        assert_allclose(uy, 0.0, atol=1e-12)


def test_von_mises_stress_matches_uniaxial_traction() -> None:
    """Under (near-)pure uniaxial stress, von Mises equivalent stress
    closely matches the axial stress -- verified on the element farthest
    from the support disturbance, where sigma_y/tau_xy are smallest (but
    not exactly zero, so this comparison is close, not exact).
    """
    result, mesh, _ = _build_result()

    element_ids = [element.id for element in mesh.elements]
    von_mises = result.element_von_mises(element_ids[-1])
    assert_allclose(von_mises, TRACTION, rtol=2e-4)
