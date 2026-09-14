"""Tests for femtoolkit.postprocessing.post_processor."""

import numpy as np
import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.postprocessing.post_processor import PostProcessor
from femtoolkit.postprocessing.result_model import MeshTopology, ResultStep, SimulationResult

_HEX8_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def _topology() -> MeshTopology:
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_HEX8_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)
    return MeshTopology.from_mesh(mesh)


def _single_step_result(topology: MeshTopology) -> SimulationResult:
    temperatures = {node_id: 300.0 + node_id * 10.0 for node_id in topology.node_ids}
    flux = {1: np.array([30.0, 40.0, 0.0])}
    step = ResultStep(
        index=0,
        time=0.0,
        nodal_fields={"temperature": temperatures},
        element_fields={"heat_flux": flux},
    )
    return SimulationResult(topology, steps=(step,))


def _multi_step_result(topology: MeshTopology) -> SimulationResult:
    steps = tuple(
        ResultStep(
            index=i,
            time=float(i),
            nodal_fields={"temperature": {1: 300.0 + 10.0 * i, 2: 310.0 + 10.0 * i}},
            element_fields={"stress": {1: np.array([1.0e6 * i, 0.0, 0.0, 0.0, 0.0, 0.0])}},
        )
        for i in range(4)
    )
    return SimulationResult(topology, steps=steps)


def test_minimum_maximum_mean_range_for_scalar_nodal_field() -> None:
    topology = _topology()
    result = _single_step_result(topology)
    pp = PostProcessor(result)

    values = list(result.final_step.nodal_field("temperature").values())
    assert pp.minimum("temperature") == pytest.approx(min(values))
    assert pp.maximum("temperature") == pytest.approx(max(values))
    assert pp.mean("temperature") == pytest.approx(sum(values) / len(values))
    assert pp.value_range("temperature") == pytest.approx((min(values), max(values)))


def test_maximum_element_field_uses_magnitude_by_default() -> None:
    topology = _topology()
    result = _single_step_result(topology)
    pp = PostProcessor(result)
    assert pp.maximum("heat_flux", kind="element") == pytest.approx(50.0)


def test_maximum_with_explicit_component() -> None:
    topology = _topology()
    result = _single_step_result(topology)
    pp = PostProcessor(result)
    assert pp.maximum("heat_flux", kind="element", component=1) == pytest.approx(40.0)


def test_field_values_kind_validation() -> None:
    topology = _topology()
    result = _single_step_result(topology)
    pp = PostProcessor(result)
    with pytest.raises(ValidationError):
        pp.maximum("temperature", kind="bogus")


def test_nodal_values_and_element_values() -> None:
    topology = _topology()
    result = _single_step_result(topology)
    pp = PostProcessor(result)
    assert pp.nodal_values("temperature") == result.final_step.nodal_field("temperature")
    assert set(pp.element_values("heat_flux")) == {1}


def test_values_at_nodes_subset() -> None:
    topology = _topology()
    result = _single_step_result(topology)
    pp = PostProcessor(result)
    subset = pp.values_at_nodes("temperature", [1, 2])
    assert set(subset) == {1, 2}
    assert subset[1] == pytest.approx(310.0)


def test_history_for_node() -> None:
    topology = _topology()
    result = _multi_step_result(topology)
    pp = PostProcessor(result)
    history = pp.history("temperature", node_id=1)
    assert np.array_equal(history, [300.0, 310.0, 320.0, 330.0])


def test_history_for_element_with_component() -> None:
    topology = _topology()
    result = _multi_step_result(topology)
    pp = PostProcessor(result)
    history = pp.history("stress", element_id=1, component=0)
    assert np.array_equal(history, [0.0, 1.0e6, 2.0e6, 3.0e6])


def test_history_requires_exactly_one_of_node_or_element() -> None:
    topology = _topology()
    result = _multi_step_result(topology)
    pp = PostProcessor(result)
    with pytest.raises(ValidationError):
        pp.history("temperature")
    with pytest.raises(ValidationError):
        pp.history("temperature", node_id=1, element_id=1)


def test_summary_delegates_to_field_calculator() -> None:
    topology = _topology()
    result = _multi_step_result(topology)
    pp = PostProcessor(result)
    summary = pp.summary()
    assert summary.num_steps == 4
    assert summary.maximum_temperature == pytest.approx(340.0)
