"""Validation Cases 4 and 5: force equilibrium and reaction equilibrium.

**Force equilibrium (Case 4).** The equivalent nodal forces produced by
a distributed traction must sum to exactly the traction's total physical
force, regardless of mesh density or element type (CST vs. Q4):

.. code-block:: text

    sum(equivalent nodal forces) = traction * boundary_length * thickness

**Reaction equilibrium (Case 5).** For a solved, properly constrained
model, the sum of every reaction plus the sum of every applied force
(here, the distributed load's equivalent nodal forces) must vanish --
global static equilibrium, computed via the existing ``R = [K]{u} - {F}``
result system with no changes needed for Version 9:

.. code-block:: text

    sum(reactions) + sum(applied forces) ~= 0
"""

import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import (
    DistributedLoad,
    LoadCase,
    TranslationDOF,
    distributed_load_to_nodal_loads,
)
from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import create_quad_mesh, create_triangular_mesh

YOUNGS_MODULUS = 200e9
POISSON_RATIO = 0.3
THICKNESS = 0.01
WIDTH = 2.0
HEIGHT = 1.0
TRACTION = 500e3


@pytest.fixture
def material() -> LinearElastic2D:
    return LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, formulation="plane_stress"
    )


@pytest.fixture
def domain() -> Rectangle:
    return Rectangle(width=WIDTH, height=HEIGHT)


# --- Case 4: force equilibrium (no solve required) ---


@pytest.mark.parametrize("nx,ny", [(2, 1), (4, 2), (6, 3)])
def test_force_equilibrium_quad_mesh_various_densities(
    material: LinearElastic2D, domain: Rectangle, nx: int, ny: int
) -> None:
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=nx, ny=ny, material=material, thickness=THICKNESS
    )
    load = DistributedLoad(domain.boundary("top"), magnitude=TRACTION, direction="normal")

    nodal_loads = distributed_load_to_nodal_loads(mesh, load)
    total_fy = sum(nl.value for nl in nodal_loads if nl.dof == TranslationDOF.Y)

    expected = TRACTION * WIDTH * THICKNESS
    assert_allclose(total_fy, expected, rtol=1e-9)


def test_force_equilibrium_triangular_mesh(material: LinearElastic2D, domain: Rectangle) -> None:
    mesh = create_triangular_mesh(
        width=WIDTH, height=HEIGHT, nx=4, ny=2, material=material, thickness=THICKNESS
    )
    load = DistributedLoad(domain.boundary("bottom"), magnitude=TRACTION, direction="normal")

    nodal_loads = distributed_load_to_nodal_loads(mesh, load)
    total_fy = sum(nl.value for nl in nodal_loads if nl.dof == TranslationDOF.Y)

    expected = -TRACTION * WIDTH * THICKNESS  # bottom's outward normal is -Y
    assert_allclose(total_fy, expected, rtol=1e-9)


def test_force_equilibrium_tangential_traction(
    material: LinearElastic2D, domain: Rectangle
) -> None:
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=4, ny=2, material=material, thickness=THICKNESS
    )
    load = DistributedLoad(domain.boundary("right"), magnitude=TRACTION, direction="tangential")

    nodal_loads = distributed_load_to_nodal_loads(mesh, load)
    total_fx = sum(nl.value for nl in nodal_loads if nl.dof == TranslationDOF.X)
    total_fy = sum(nl.value for nl in nodal_loads if nl.dof == TranslationDOF.Y)

    # Tangential on the right edge (outward normal +X) points +Y.
    assert_allclose(total_fx, 0.0, atol=1e-6)
    assert_allclose(total_fy, TRACTION * HEIGHT * THICKNESS, rtol=1e-9)


# --- Case 5: reaction equilibrium (requires solving) ---


def test_reaction_equilibrium_quad_mesh(material: LinearElastic2D, domain: Rectangle) -> None:
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=4, ny=2, material=material, thickness=THICKNESS
    )

    load_case = LoadCase(name="Tension", mesh=mesh)
    load_case.fix_boundary(domain.boundary("left"), ux=0.0, uy=0.0)
    load_case.add_distributed_load(DistributedLoad(domain.boundary("right"), magnitude=TRACTION))
    result = load_case.solve()

    total_rx = sum(result.node_reaction(n.id)[0] for n in mesh.nodes)
    total_ry = sum(result.node_reaction(n.id)[1] for n in mesh.nodes)
    total_applied_fx = TRACTION * HEIGHT * THICKNESS

    assert_allclose(total_rx + total_applied_fx, 0.0, atol=1e-6)
    assert_allclose(total_ry, 0.0, atol=1e-6)


def test_reaction_equilibrium_triangular_mesh(material: LinearElastic2D, domain: Rectangle) -> None:
    mesh = create_triangular_mesh(
        width=WIDTH, height=HEIGHT, nx=4, ny=2, material=material, thickness=THICKNESS
    )

    load_case = LoadCase(name="Tension", mesh=mesh)
    load_case.fix_boundary(domain.boundary("left"), ux=0.0, uy=0.0)
    load_case.add_distributed_load(DistributedLoad(domain.boundary("right"), magnitude=TRACTION))
    result = load_case.solve()

    total_rx = sum(result.node_reaction(n.id)[0] for n in mesh.nodes)
    total_ry = sum(result.node_reaction(n.id)[1] for n in mesh.nodes)
    total_applied_fx = TRACTION * HEIGHT * THICKNESS

    assert_allclose(total_rx + total_applied_fx, 0.0, atol=1e-6)
    assert_allclose(total_ry, 0.0, atol=1e-6)


def test_reaction_equilibrium_with_pin_and_roller_support(
    material: LinearElastic2D, domain: Rectangle
) -> None:
    """A classical statically determinate 2D support scheme -- a pin
    (ux=uy=0) at one corner plus a roller (ux=0 only) at another -- must
    also close equilibrium exactly. This is a different, independent
    constraint pattern from the "fix every left node" scheme used
    elsewhere, checking that equilibrium holds generally, not only for
    one particular (over-constrained) support layout.

    Fixing ``uy=0`` at every left-edge node (all sharing ``x=0``) does
    NOT, by itself, prevent rigid-body rotation about a point on that
    same line -- under a small rotation about any point with ``x=0``,
    every left-edge node's Y-displacement is identically zero regardless
    of the rotation angle, so that constraint alone cannot detect it.
    Preventing rotation requires constraining X (or Y) at two points that
    are NOT both admissible pivots for the same free rotation -- exactly
    what the pin + roller pairing below provides.
    """
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=4, ny=2, material=material, thickness=THICKNESS
    )

    left_nodes = mesh.nodes_on_boundary(domain.boundary("left"))
    pin_node = min(left_nodes, key=lambda node: node.y)  # bottom-left corner
    roller_node = max(left_nodes, key=lambda node: node.y)  # top-left corner

    from femtoolkit.analysis import BoundaryCondition

    load_case = LoadCase(name="Tension", mesh=mesh)
    load_case.add_boundary_condition(BoundaryCondition(pin_node.id, TranslationDOF.X, 0.0))
    load_case.add_boundary_condition(BoundaryCondition(pin_node.id, TranslationDOF.Y, 0.0))
    load_case.add_boundary_condition(BoundaryCondition(roller_node.id, TranslationDOF.X, 0.0))
    load_case.add_distributed_load(DistributedLoad(domain.boundary("right"), magnitude=TRACTION))
    result = load_case.solve()

    total_rx = sum(result.node_reaction(n.id)[0] for n in mesh.nodes)
    total_ry = sum(result.node_reaction(n.id)[1] for n in mesh.nodes)
    total_applied_fx = TRACTION * HEIGHT * THICKNESS

    assert_allclose(total_rx + total_applied_fx, 0.0, atol=1e-6)
    assert_allclose(total_ry, 0.0, atol=1e-6)
