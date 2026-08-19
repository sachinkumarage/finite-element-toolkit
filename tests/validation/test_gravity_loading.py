"""Validation: gravity loading produces the correct total structural weight.

For a cantilevered plate under only its own weight (fixed left edge, no
other loads), the total vertical reaction must equal the plate's total
weight exactly:

.. code-block:: text

    W = density * (width * height) * thickness * g

This is a pure statement of vertical equilibrium (Newton's second law,
static case) and must hold regardless of mesh density or element type
(CST vs. Q4) -- an independent check on the gravity-to-nodal-force
conversion in :mod:`femtoolkit.analysis.body_load`, distinct from the
per-node weight-split unit tests in ``tests/test_body_load.py``.
"""

import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import DistributedLoad, GravityLoad, LoadCase
from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import create_quad_mesh, create_triangular_mesh

YOUNGS_MODULUS = 200e9
POISSON_RATIO = 0.3
DENSITY = 7850.0
THICKNESS = 0.01
WIDTH = 2.0
HEIGHT = 1.0
G = 9.81


@pytest.fixture
def material() -> LinearElastic2D:
    return LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS,
        poisson_ratio=POISSON_RATIO,
        formulation="plane_stress",
        density=DENSITY,
    )


@pytest.fixture
def domain() -> Rectangle:
    return Rectangle(width=WIDTH, height=HEIGHT)


@pytest.mark.parametrize("nx,ny", [(2, 1), (4, 2), (6, 3)])
def test_quad_mesh_total_reaction_equals_total_weight(
    material: LinearElastic2D, domain: Rectangle, nx: int, ny: int
) -> None:
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=nx, ny=ny, material=material, thickness=THICKNESS
    )
    load_case = LoadCase(name="Self Weight", mesh=mesh)
    load_case.fix_boundary(domain.boundary("left"), ux=0.0, uy=0.0)
    load_case.add_gravity_load(GravityLoad(g=G))
    result = load_case.solve()

    total_ry = sum(result.node_reaction(n.id)[1] for n in mesh.nodes)
    total_rx = sum(result.node_reaction(n.id)[0] for n in mesh.nodes)
    expected_weight = DENSITY * (WIDTH * HEIGHT) * THICKNESS * G

    assert_allclose(total_ry, expected_weight, rtol=1e-9)
    assert_allclose(total_rx, 0.0, atol=1e-6)


def test_triangular_mesh_total_reaction_equals_total_weight(
    material: LinearElastic2D, domain: Rectangle
) -> None:
    mesh = create_triangular_mesh(
        width=WIDTH, height=HEIGHT, nx=4, ny=2, material=material, thickness=THICKNESS
    )
    load_case = LoadCase(name="Self Weight", mesh=mesh)
    load_case.fix_boundary(domain.boundary("left"), ux=0.0, uy=0.0)
    load_case.add_gravity_load(GravityLoad(g=G))
    result = load_case.solve()

    total_ry = sum(result.node_reaction(n.id)[1] for n in mesh.nodes)
    expected_weight = DENSITY * (WIDTH * HEIGHT) * THICKNESS * G

    assert_allclose(total_ry, expected_weight, rtol=1e-9)


def test_gravity_combined_with_distributed_load_still_balances(
    material: LinearElastic2D, domain: Rectangle
) -> None:
    """Gravity plus an unrelated horizontal traction: reactions must
    balance the sum of both, independently, in their own directions.
    """
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=4, ny=2, material=material, thickness=THICKNESS
    )
    traction = 5000.0

    load_case = LoadCase(name="Combined", mesh=mesh)
    load_case.fix_boundary(domain.boundary("left"), ux=0.0, uy=0.0)
    load_case.add_gravity_load(GravityLoad(g=G))
    load_case.add_distributed_load(DistributedLoad(domain.boundary("right"), magnitude=traction))
    result = load_case.solve()

    total_rx = sum(result.node_reaction(n.id)[0] for n in mesh.nodes)
    total_ry = sum(result.node_reaction(n.id)[1] for n in mesh.nodes)
    expected_weight = DENSITY * (WIDTH * HEIGHT) * THICKNESS * G
    expected_axial_force = traction * HEIGHT * THICKNESS

    assert_allclose(total_ry, expected_weight, rtol=1e-9)
    assert_allclose(total_rx + expected_axial_force, 0.0, atol=1e-6)
