"""Tests for femtoolkit.postprocessing.result_model."""

import numpy as np
import pytest

from femtoolkit.exceptions import EntityNotFoundError, ValidationError
from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.postprocessing.result_model import MeshTopology, ResultStep, SimulationResult

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def _hex8_mesh() -> tuple[Mesh, Hex8Element3D]:
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)
    return mesh, hexa


def test_mesh_topology_from_mesh_captures_nodes_and_elements() -> None:
    mesh, hexa = _hex8_mesh()
    topology = MeshTopology.from_mesh(mesh)

    assert set(topology.node_ids) == {1, 2, 3, 4, 5, 6, 7, 8}
    assert topology.node_coordinates[1] == (0.0, 0.0, 0.0)
    assert topology.element_ids == (1,)
    assert topology.element_connectivity[1] == tuple(node.id for node in hexa.nodes)
    assert topology.element_types[1] == "Hex8Element3D"


def test_mesh_topology_node_coordinate_array_matches_node_ids_order() -> None:
    mesh, _hexa = _hex8_mesh()
    topology = MeshTopology.from_mesh(mesh)
    array = topology.node_coordinate_array()

    assert array.shape == (8, 3)
    for index, node_id in enumerate(topology.node_ids):
        assert tuple(array[index]) == topology.node_coordinates[node_id]


def test_mesh_topology_from_elements_derives_nodes_from_connectivity() -> None:
    _mesh, hexa = _hex8_mesh()
    topology = MeshTopology.from_elements([hexa])

    assert set(topology.node_ids) == {node.id for node in hexa.nodes}
    assert topology.element_ids == (1,)


def test_mesh_topology_rejects_inconsistent_node_data() -> None:
    with pytest.raises(ValidationError):
        MeshTopology(
            node_ids=(1, 2),
            node_coordinates={1: (0.0, 0.0, 0.0)},
            element_ids=(),
            element_connectivity={},
            element_types={},
        )


def test_mesh_topology_rejects_inconsistent_element_data() -> None:
    with pytest.raises(ValidationError):
        MeshTopology(
            node_ids=(1,),
            node_coordinates={1: (0.0, 0.0, 0.0)},
            element_ids=(1,),
            element_connectivity={1: (1,)},
            element_types={},
        )


def test_result_step_stores_and_retrieves_scalar_nodal_field() -> None:
    step = ResultStep(index=0, time=0.0, nodal_fields={"temperature": {1: 300.0, 2: 310.0}})
    assert step.nodal_value("temperature", 1) == 300.0
    assert step.nodal_field("temperature") == {1: 300.0, 2: 310.0}


def test_result_step_stores_and_retrieves_vector_element_field() -> None:
    flux = np.array([1.0, 2.0, 3.0])
    step = ResultStep(index=0, time=0.0, element_fields={"heat_flux": {1: flux}})
    assert np.array_equal(step.element_value("heat_flux", 1), flux)


def test_result_step_raises_for_missing_field_name() -> None:
    step = ResultStep(index=0, time=0.0, nodal_fields={"temperature": {1: 300.0}})
    with pytest.raises(ValidationError):
        step.nodal_field("displacement")
    with pytest.raises(ValidationError):
        step.element_field("stress")


def test_result_step_raises_for_missing_entity() -> None:
    step = ResultStep(index=0, time=0.0, nodal_fields={"temperature": {1: 300.0}})
    with pytest.raises(EntityNotFoundError):
        step.nodal_value("temperature", 999)


def test_simulation_result_requires_at_least_one_step() -> None:
    mesh, _hexa = _hex8_mesh()
    topology = MeshTopology.from_mesh(mesh)
    with pytest.raises(ValidationError):
        SimulationResult(topology, steps=())


def test_simulation_result_final_step_and_step_indexing() -> None:
    mesh, _hexa = _hex8_mesh()
    topology = MeshTopology.from_mesh(mesh)
    step0 = ResultStep(index=0, time=0.0, nodal_fields={"temperature": {1: 300.0}})
    step1 = ResultStep(index=1, time=1.0, nodal_fields={"temperature": {1: 310.0}})
    result = SimulationResult(topology, steps=(step0, step1))

    assert result.num_steps == 2
    assert result.final_step is step1
    assert result.step(0) is step0
    assert result.step(-1) is step1


def test_simulation_result_times_array() -> None:
    mesh, _hexa = _hex8_mesh()
    topology = MeshTopology.from_mesh(mesh)
    steps = tuple(
        ResultStep(index=i, time=float(i) * 0.5, nodal_fields={"temperature": {1: 300.0 + i}})
        for i in range(4)
    )
    result = SimulationResult(topology, steps=steps)
    assert np.array_equal(result.times, [0.0, 0.5, 1.0, 1.5])


def test_simulation_result_nodal_history_scalar() -> None:
    mesh, _hexa = _hex8_mesh()
    topology = MeshTopology.from_mesh(mesh)
    steps = tuple(
        ResultStep(index=i, time=float(i), nodal_fields={"temperature": {1: 300.0 + 10.0 * i}})
        for i in range(3)
    )
    result = SimulationResult(topology, steps=steps)
    assert np.array_equal(result.nodal_history("temperature", 1), [300.0, 310.0, 320.0])


def test_simulation_result_nodal_history_vector_component() -> None:
    mesh, _hexa = _hex8_mesh()
    topology = MeshTopology.from_mesh(mesh)
    steps = tuple(
        ResultStep(
            index=i, time=float(i), nodal_fields={"displacement": {1: np.array([i * 1.0, i * 2.0])}}
        )
        for i in range(3)
    )
    result = SimulationResult(topology, steps=steps)
    assert np.array_equal(result.nodal_history("displacement", 1, component=1), [0.0, 2.0, 4.0])


def test_simulation_result_element_history() -> None:
    mesh, hexa = _hex8_mesh()
    topology = MeshTopology.from_mesh(mesh)
    steps = tuple(
        ResultStep(
            index=i,
            time=float(i),
            element_fields={"heat_flux": {hexa.id: np.array([i, 0.0, 0.0])}},
        )
        for i in range(3)
    )
    result = SimulationResult(topology, steps=steps)
    history = result.element_history("heat_flux", hexa.id, component=0)
    assert np.array_equal(history, [0.0, 1.0, 2.0])
