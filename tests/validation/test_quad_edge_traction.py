"""Validation Case 2: constant traction on a single Q4 edge.

Geometry: a single Q4 element, nodes (0,0), (1,0), (1,1), (0,1), with a
constant traction applied to its bottom edge (node 1 to node 2, length
1 m). E = 200 GPa, v = 0.3, thickness = 0.1 m, traction = 1000 Pa.

Same analytical solution and split as the CST case (Validation Case 1) --
the two element types share the exact same 2-node straight-edge
formulation, since neither has midside nodes -- verified here to confirm
Q4 edges are handled identically to CST edges by the boundary-load
machinery, and specifically that a Q4 element's *bilinear* interior shape
functions do not affect its (still linear) *edge* shape functions.

.. code-block:: text

    F_total = traction * length * thickness = 1000 * 1 * 0.1 = 100 N
    F_node  = F_total / 2 = 50 N
"""

from numpy.testing import assert_allclose

from femtoolkit.analysis import DistributedLoad, TranslationDOF, distributed_load_to_nodal_loads
from femtoolkit.continuum.edge import edge_equivalent_nodal_force
from femtoolkit.geometry import BoundaryRegion, LineSegment2D, Point2D
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, Node, QuadElement2D

THICKNESS = 0.1
TRACTION = 1000.0
EDGE_LENGTH = 1.0
EXPECTED_TOTAL_FORCE = TRACTION * EDGE_LENGTH * THICKNESS
EXPECTED_NODE_FORCE = EXPECTED_TOTAL_FORCE / 2.0


def _build_mesh_and_region():
    material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=1.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=1.0, z=0.0)
    element = QuadElement2D(
        id=1, nodes=(node_1, node_2, node_3, node_4), material=material, thickness=THICKNESS
    )

    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4):
        mesh.add_node(node)
    mesh.add_element(element)

    bottom_edge = BoundaryRegion(
        "bottom", LineSegment2D(Point2D(0.0, 0.0), Point2D(1.0, 0.0)), outward_normal=(0.0, -1.0)
    )
    return mesh, bottom_edge


def test_quad_edge_nodal_forces_match_analytical_split() -> None:
    mesh, bottom_edge = _build_mesh_and_region()
    load = DistributedLoad(bottom_edge, magnitude=TRACTION, direction="normal")

    nodal_loads = distributed_load_to_nodal_loads(mesh, load)
    fy_by_node = {nl.node_id: nl.value for nl in nodal_loads if nl.dof == TranslationDOF.Y}

    assert_allclose(fy_by_node[1], -EXPECTED_NODE_FORCE, rtol=1e-9)
    assert_allclose(fy_by_node[2], -EXPECTED_NODE_FORCE, rtol=1e-9)


def test_quad_edge_total_force_matches_traction_times_length_times_thickness() -> None:
    mesh, bottom_edge = _build_mesh_and_region()
    load = DistributedLoad(bottom_edge, magnitude=TRACTION, direction="normal")

    nodal_loads = distributed_load_to_nodal_loads(mesh, load)
    total_fy = sum(nl.value for nl in nodal_loads if nl.dof == TranslationDOF.Y)

    assert_allclose(total_fy, -EXPECTED_TOTAL_FORCE, rtol=1e-9)


def test_quad_edge_matches_cst_edge_result_for_the_same_geometry() -> None:
    """A Q4 element's edge and a CST element's edge, given the same two
    endpoint coordinates, must produce identical equivalent nodal forces.
    """
    mesh, bottom_edge = _build_mesh_and_region()
    load = DistributedLoad(bottom_edge, magnitude=TRACTION, direction="normal")
    quad_loads = distributed_load_to_nodal_loads(mesh, load)
    quad_fy = {nl.node_id: nl.value for nl in quad_loads if nl.dof == TranslationDOF.Y}

    direct_force = edge_equivalent_nodal_force((0.0, 0.0), (1.0, 0.0), (0.0, -TRACTION), THICKNESS)
    assert_allclose([quad_fy[1], quad_fy[2]], [direct_force[1], direct_force[3]], rtol=1e-9)


def test_quad_element_unloaded_edges_contribute_no_force() -> None:
    """Only the bottom edge is on the boundary region; the top edge (also
    a boundary edge, but not part of `bottom_edge`) must contribute
    nothing.
    """
    mesh, bottom_edge = _build_mesh_and_region()
    load = DistributedLoad(bottom_edge, magnitude=TRACTION, direction="normal")

    nodal_loads = distributed_load_to_nodal_loads(mesh, load)
    loaded_node_ids = {nl.node_id for nl in nodal_loads}
    assert loaded_node_ids == {1, 2}
