"""Tests for DistributedLoad and its conversion to nodal loads."""

import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import TranslationDOF
from femtoolkit.analysis.distributed_load import DistributedLoad, distributed_load_to_nodal_loads
from femtoolkit.exceptions import ValidationError
from femtoolkit.geometry import Point2D, Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import create_quad_mesh, create_triangular_mesh

WIDTH = 2.0
HEIGHT = 1.0
THICKNESS = 0.01


@pytest.fixture
def material() -> LinearElastic2D:
    return LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")


@pytest.fixture
def domain() -> Rectangle:
    return Rectangle(width=WIDTH, height=HEIGHT)


# --- DistributedLoad.traction_vector() ---


def test_normal_direction_uses_outward_normal(domain: Rectangle) -> None:
    load = DistributedLoad(domain.boundary("right"), magnitude=1000.0, direction="normal")

    assert load.traction_vector() == (1000.0, 0.0)


def test_normal_direction_on_left_boundary_points_outward(domain: Rectangle) -> None:
    load = DistributedLoad(domain.boundary("left"), magnitude=1000.0, direction="normal")

    assert load.traction_vector() == (-1000.0, 0.0)


def test_tangential_direction(domain: Rectangle) -> None:
    load = DistributedLoad(domain.boundary("right"), magnitude=1000.0, direction="tangential")

    tx, ty = load.traction_vector()
    assert_allclose((tx, ty), (0.0, 1000.0))


def test_global_x_direction(domain: Rectangle) -> None:
    load = DistributedLoad(domain.boundary("top"), magnitude=500.0, direction="global_x")

    assert load.traction_vector() == (500.0, 0.0)


def test_global_y_direction(domain: Rectangle) -> None:
    load = DistributedLoad(domain.boundary("top"), magnitude=-500.0, direction="global_y")

    assert load.traction_vector() == (0.0, -500.0)


def test_global_direction_uses_magnitude_directly(domain: Rectangle) -> None:
    load = DistributedLoad(domain.boundary("top"), magnitude=(100.0, -200.0), direction="global")

    assert load.traction_vector() == (100.0, -200.0)


def test_default_direction_is_normal(domain: Rectangle) -> None:
    load = DistributedLoad(domain.boundary("right"), magnitude=1000.0)

    assert load.direction == "normal"


def test_invalid_direction_raises(domain: Rectangle) -> None:
    with pytest.raises(ValidationError):
        DistributedLoad(domain.boundary("right"), magnitude=1000.0, direction="diagonal")  # type: ignore[arg-type]


def test_global_direction_requires_tuple_magnitude(domain: Rectangle) -> None:
    with pytest.raises(ValidationError):
        DistributedLoad(domain.boundary("right"), magnitude=1000.0, direction="global")


def test_non_global_direction_requires_scalar_magnitude(domain: Rectangle) -> None:
    with pytest.raises(ValidationError):
        DistributedLoad(domain.boundary("right"), magnitude=(1000.0, 0.0), direction="normal")


# --- distributed_load_to_nodal_loads: force equilibrium ---


def test_total_equivalent_force_matches_traction_times_length_times_thickness(
    material: LinearElastic2D, domain: Rectangle
) -> None:
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=4, ny=2, material=material, thickness=THICKNESS
    )
    load = DistributedLoad(domain.boundary("right"), magnitude=1000.0, direction="normal")

    nodal_loads = distributed_load_to_nodal_loads(mesh, load)

    total_fx = sum(nl.value for nl in nodal_loads if nl.dof == TranslationDOF.X)
    total_fy = sum(nl.value for nl in nodal_loads if nl.dof == TranslationDOF.Y)
    expected_fx = 1000.0 * HEIGHT * THICKNESS
    assert_allclose(total_fx, expected_fx, rtol=1e-9)
    assert_allclose(total_fy, 0.0, atol=1e-9)


def test_total_equivalent_force_independent_of_mesh_density(
    material: LinearElastic2D, domain: Rectangle
) -> None:
    """A consistent nodal-load distribution must sum to the same total
    force regardless of how finely the boundary happens to be meshed.
    """
    load = DistributedLoad(domain.boundary("right"), magnitude=1000.0, direction="normal")
    expected_fx = 1000.0 * HEIGHT * THICKNESS

    for nx, ny in [(2, 1), (4, 2), (8, 4)]:
        mesh = create_quad_mesh(
            width=WIDTH, height=HEIGHT, nx=nx, ny=ny, material=material, thickness=THICKNESS
        )
        nodal_loads = distributed_load_to_nodal_loads(mesh, load)
        total_fx = sum(nl.value for nl in nodal_loads if nl.dof == TranslationDOF.X)
        assert_allclose(total_fx, expected_fx, rtol=1e-9)


def test_distributed_load_on_triangular_mesh(material: LinearElastic2D, domain: Rectangle) -> None:
    mesh = create_triangular_mesh(
        width=WIDTH, height=HEIGHT, nx=4, ny=2, material=material, thickness=THICKNESS
    )
    load = DistributedLoad(domain.boundary("right"), magnitude=1000.0, direction="normal")

    nodal_loads = distributed_load_to_nodal_loads(mesh, load)

    total_fx = sum(nl.value for nl in nodal_loads if nl.dof == TranslationDOF.X)
    assert_allclose(total_fx, 1000.0 * HEIGHT * THICKNESS, rtol=1e-9)


def test_corner_nodes_get_half_the_contribution_of_shared_interior_nodes(
    material: LinearElastic2D, domain: Rectangle
) -> None:
    """For a uniformly subdivided boundary, an interior boundary node is
    shared by two edges and gets roughly twice a corner node's force.
    """
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=4, ny=2, material=material, thickness=THICKNESS
    )
    load = DistributedLoad(domain.boundary("right"), magnitude=1000.0, direction="normal")

    nodal_loads = distributed_load_to_nodal_loads(mesh, load)
    fx_by_node: dict[int, float] = {}
    for nl in nodal_loads:
        if nl.dof == TranslationDOF.X:
            fx_by_node[nl.node_id] = fx_by_node.get(nl.node_id, 0.0) + nl.value

    right_node_ids = [n.id for n in mesh.nodes_on_boundary(domain.boundary("right"))]
    corner_force = fx_by_node[right_node_ids[0]]  # bottom-right corner
    interior_force = fx_by_node[right_node_ids[1]]  # first interior node up the edge
    assert_allclose(interior_force, 2.0 * corner_force, rtol=1e-9)


def test_no_loads_when_boundary_has_no_matching_edges(
    material: LinearElastic2D, domain: Rectangle
) -> None:
    """A boundary region with no coincident mesh nodes produces no loads."""
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=2, ny=2, material=material, thickness=0.01
    )
    far_away_domain = Rectangle(width=WIDTH, height=HEIGHT, origin=Point2D(100.0, 100.0))

    load = DistributedLoad(far_away_domain.boundary("left"), magnitude=1000.0)
    assert distributed_load_to_nodal_loads(mesh, load) == []
