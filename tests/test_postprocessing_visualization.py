"""Tests for femtoolkit.postprocessing.visualization.

Forces the non-interactive "Agg" backend before importing matplotlib's
pyplot machinery indirectly through the visualization module, so these
tests run headlessly regardless of the environment.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from femtoolkit.exceptions import ValidationError
from femtoolkit.postprocessing.result_model import MeshTopology, ResultStep, SimulationResult
from femtoolkit.postprocessing.visualization import (
    plot_deformed_shape_2d,
    plot_element_scatter_2d,
    plot_heat_flux_vectors_2d,
    plot_line,
    plot_nodal_contour_2d,
    plot_time_history,
)

_QUAD_COORDS_2D = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]


def _quad_topology() -> MeshTopology:
    node_ids = tuple(range(1, 5))
    node_coordinates = {node_id: _QUAD_COORDS_2D[i] for i, node_id in enumerate(node_ids)}
    return MeshTopology(
        node_ids=node_ids,
        node_coordinates=node_coordinates,
        element_ids=(1,),
        element_connectivity={1: node_ids},
        element_types={1: "QuadElement2D"},
    )


def test_plot_line_returns_figure() -> None:
    figure = plot_line([0, 1, 2], [10, 20, 15], xlabel="x", ylabel="y", title="Test")
    assert isinstance(figure, Figure)
    assert figure.axes[0].get_title() == "Test"


def test_plot_nodal_contour_2d_returns_figure_with_colorbar() -> None:
    topology = _quad_topology()
    temperatures = {1: 300.0, 2: 320.0, 3: 340.0, 4: 310.0}
    step = ResultStep(index=0, time=0.0, nodal_fields={"temperature": temperatures})
    result = SimulationResult(topology, steps=(step,), field_units={"temperature": "K"})

    figure = plot_nodal_contour_2d(result, "temperature")
    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2  # main axes + colorbar axes


def test_plot_nodal_contour_2d_with_component() -> None:
    topology = _quad_topology()
    displacement = {node_id: np.array([0.01 * node_id, 0.0]) for node_id in topology.node_ids}
    step = ResultStep(index=0, time=0.0, nodal_fields={"displacement": displacement})
    result = SimulationResult(topology, steps=(step,))

    figure = plot_nodal_contour_2d(result, "displacement", component=0)
    assert isinstance(figure, Figure)


def test_plot_element_scatter_2d_returns_figure() -> None:
    topology = _quad_topology()
    step = ResultStep(index=0, time=0.0, element_fields={"von_mises_stress": {1: 150e6}})
    result = SimulationResult(topology, steps=(step,), field_units={"von_mises_stress": "Pa"})

    figure = plot_element_scatter_2d(result, "von_mises_stress")
    assert isinstance(figure, Figure)


def test_plot_heat_flux_vectors_2d_returns_figure() -> None:
    topology = _quad_topology()
    step = ResultStep(index=0, time=0.0, element_fields={"heat_flux": {1: np.array([50.0, 0.0])}})
    result = SimulationResult(topology, steps=(step,), field_units={"heat_flux": "W/m^2"})

    figure = plot_heat_flux_vectors_2d(result)
    assert isinstance(figure, Figure)


def test_plot_heat_flux_vectors_2d_requires_heat_flux_field() -> None:
    topology = _quad_topology()
    step = ResultStep(index=0, time=0.0, nodal_fields={"temperature": {1: 300.0}})
    result = SimulationResult(topology, steps=(step,))
    with pytest.raises(ValidationError):
        plot_heat_flux_vectors_2d(result)


def test_plot_deformed_shape_2d_without_color_field() -> None:
    topology = _quad_topology()
    displacement = {node_id: np.array([0.01, 0.0]) for node_id in topology.node_ids}
    step = ResultStep(index=0, time=0.0, nodal_fields={"displacement": displacement})
    result = SimulationResult(topology, steps=(step,))

    figure = plot_deformed_shape_2d(result, scale=10.0)
    assert isinstance(figure, Figure)


def test_plot_deformed_shape_2d_with_color_field() -> None:
    topology = _quad_topology()
    displacement = {node_id: np.array([0.01, 0.0]) for node_id in topology.node_ids}
    step = ResultStep(
        index=0,
        time=0.0,
        nodal_fields={"displacement": displacement},
        element_fields={"von_mises_stress": {1: 100e6}},
    )
    result = SimulationResult(topology, steps=(step,))

    figure = plot_deformed_shape_2d(result, scale=10.0, color_field="von_mises_stress")
    assert isinstance(figure, Figure)


def test_plot_time_history_for_node() -> None:
    topology = _quad_topology()
    steps = tuple(
        ResultStep(index=i, time=float(i), nodal_fields={"temperature": {1: 300.0 + i}})
        for i in range(3)
    )
    result = SimulationResult(topology, steps=steps)

    figure = plot_time_history(result, "temperature", node_id=1)
    assert isinstance(figure, Figure)
    line = figure.axes[0].lines[0]
    assert np.array_equal(line.get_ydata(), [300.0, 301.0, 302.0])


def test_plot_time_history_for_element() -> None:
    topology = _quad_topology()
    steps = tuple(
        ResultStep(index=i, time=float(i), element_fields={"heat_flux": {1: np.array([i, 0.0])}})
        for i in range(3)
    )
    result = SimulationResult(topology, steps=steps)

    figure = plot_time_history(result, "heat_flux", element_id=1)
    assert isinstance(figure, Figure)


def test_plot_time_history_requires_exactly_one_of_node_or_element() -> None:
    topology = _quad_topology()
    step = ResultStep(index=0, time=0.0, nodal_fields={"temperature": {1: 300.0}})
    result = SimulationResult(topology, steps=(step,))
    with pytest.raises(ValidationError):
        plot_time_history(result, "temperature")
