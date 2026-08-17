"""Validation Case 1: constant traction on a single CST edge.

Geometry: a single CST triangle, nodes (0,0), (1,0), (0,1), with a
constant traction applied to its bottom edge (node 1 to node 2, length
1 m). E = 200 GPa, v = 0.3 (plane stress, elastic constants unused by
this force-only check), thickness = 0.1 m, traction = 1000 Pa (normal,
i.e. along the edge's outward normal, straight down).

Analytical solution: for a straight, uniformly loaded 2-node edge, the
equivalent nodal force splits evenly between the two edge nodes:

.. code-block:: text

    F_total = traction * length * thickness = 1000 * 1 * 0.1 = 100 N
    F_node  = F_total / 2 = 50 N

This is independently verified against
:func:`femtoolkit.continuum.edge.edge_equivalent_nodal_force` called
directly (not through the mesh/boundary machinery), confirming the two
code paths agree.
"""

from numpy.testing import assert_allclose

from femtoolkit.analysis import DistributedLoad, TranslationDOF, distributed_load_to_nodal_loads
from femtoolkit.continuum.edge import edge_equivalent_nodal_force
from femtoolkit.geometry import BoundaryRegion, LineSegment2D, Point2D
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import CSTElement2D, Mesh, Node

THICKNESS = 0.1
TRACTION = 1000.0
EDGE_LENGTH = 1.0
EXPECTED_TOTAL_FORCE = TRACTION * EDGE_LENGTH * THICKNESS
EXPECTED_NODE_FORCE = EXPECTED_TOTAL_FORCE / 2.0


def _build_mesh_and_region():
    material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=0.0, y=1.0, z=0.0)
    element = CSTElement2D(
        id=1, nodes=(node_1, node_2, node_3), material=material, thickness=THICKNESS
    )

    mesh = Mesh()
    for node in (node_1, node_2, node_3):
        mesh.add_node(node)
    mesh.add_element(element)

    bottom_edge = BoundaryRegion(
        "bottom", LineSegment2D(Point2D(0.0, 0.0), Point2D(1.0, 0.0)), outward_normal=(0.0, -1.0)
    )
    return mesh, bottom_edge


def test_cst_edge_nodal_forces_match_analytical_split() -> None:
    mesh, bottom_edge = _build_mesh_and_region()
    load = DistributedLoad(bottom_edge, magnitude=TRACTION, direction="normal")

    nodal_loads = distributed_load_to_nodal_loads(mesh, load)
    fy_by_node = {nl.node_id: nl.value for nl in nodal_loads if nl.dof == TranslationDOF.Y}

    assert_allclose(fy_by_node[1], -EXPECTED_NODE_FORCE, rtol=1e-9)
    assert_allclose(fy_by_node[2], -EXPECTED_NODE_FORCE, rtol=1e-9)


def test_cst_edge_total_force_matches_traction_times_length_times_thickness() -> None:
    mesh, bottom_edge = _build_mesh_and_region()
    load = DistributedLoad(bottom_edge, magnitude=TRACTION, direction="normal")

    nodal_loads = distributed_load_to_nodal_loads(mesh, load)
    total_fy = sum(nl.value for nl in nodal_loads if nl.dof == TranslationDOF.Y)

    assert_allclose(total_fy, -EXPECTED_TOTAL_FORCE, rtol=1e-9)


def test_cst_edge_load_matches_direct_continuum_edge_computation() -> None:
    """The mesh/boundary-driven path must agree exactly with calling the
    underlying edge-integration formula directly.
    """
    mesh, bottom_edge = _build_mesh_and_region()
    load = DistributedLoad(bottom_edge, magnitude=TRACTION, direction="normal")

    nodal_loads = distributed_load_to_nodal_loads(mesh, load)
    fy_by_node = {nl.node_id: nl.value for nl in nodal_loads if nl.dof == TranslationDOF.Y}

    direct_force = edge_equivalent_nodal_force((0.0, 0.0), (1.0, 0.0), (0.0, -TRACTION), THICKNESS)
    assert_allclose([fy_by_node[1], fy_by_node[2]], [direct_force[1], direct_force[3]], rtol=1e-9)


def test_cst_edge_no_x_component_for_pure_normal_traction() -> None:
    """The bottom edge's outward normal is purely -Y, so a normal
    traction must produce zero X-direction force.
    """
    mesh, bottom_edge = _build_mesh_and_region()
    load = DistributedLoad(bottom_edge, magnitude=TRACTION, direction="normal")

    nodal_loads = distributed_load_to_nodal_loads(mesh, load)
    total_fx = sum(nl.value for nl in nodal_loads if nl.dof == TranslationDOF.X)

    assert_allclose(total_fx, 0.0, atol=1e-9)
