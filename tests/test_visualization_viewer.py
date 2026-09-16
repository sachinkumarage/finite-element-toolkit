"""Tests for femtoolkit.postprocessing.visualization_3d.viewer.FEAViewer."""

import os

import numpy as np
import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.postprocessing.result_model import MeshTopology, ResultStep, SimulationResult
from femtoolkit.postprocessing.visualization_3d.config import ViewerConfig
from femtoolkit.postprocessing.visualization_3d.viewer import FEAViewer

_HEX8_COORDS = {
    1: (0.0, 0.0, 0.0),
    2: (1.0, 0.0, 0.0),
    3: (1.0, 1.0, 0.0),
    4: (0.0, 1.0, 0.0),
    5: (0.0, 0.0, 1.0),
    6: (1.0, 0.0, 1.0),
    7: (1.0, 1.0, 1.0),
    8: (0.0, 1.0, 1.0),
}


def _topology() -> MeshTopology:
    return MeshTopology(
        node_ids=tuple(_HEX8_COORDS),
        node_coordinates=_HEX8_COORDS,
        element_ids=(1,),
        element_connectivity={1: tuple(_HEX8_COORDS)},
        element_types={1: "Hex8Element3D"},
    )


def _known_displacement() -> dict[int, np.ndarray]:
    return {node_id: np.array([0.01, 0.0, 0.0]) for node_id in _HEX8_COORDS}


def _zero_displacement() -> dict[int, np.ndarray]:
    return {node_id: np.zeros(3) for node_id in _HEX8_COORDS}


def _result(num_steps: int = 3) -> SimulationResult:
    topology = _topology()
    steps = []
    for index in range(num_steps):
        factor = (index + 1) / num_steps
        temperature = {node_id: 300.0 + 10.0 * index for node_id in _HEX8_COORDS}
        displacement = {
            node_id: np.array([0.01 * factor, 0.0, 0.0]) for node_id in _HEX8_COORDS
        }
        von_mises = {1: 1.0e6 * factor}
        heat_flux = {1: np.array([1.0, 0.0, 0.0])}
        steps.append(
            ResultStep(
                index=index,
                time=factor,
                nodal_fields={"temperature": temperature, "displacement": displacement},
                element_fields={"von_mises_stress": von_mises, "heat_flux": heat_flux},
            )
        )
    return SimulationResult(topology=topology, steps=tuple(steps))


def _viewer(**config_kwargs) -> FEAViewer:
    config_kwargs.setdefault("off_screen", True)
    return FEAViewer(_result(), ViewerConfig(**config_kwargs))


def test_viewer_construction_defaults_to_final_step() -> None:
    viewer = _viewer()
    assert viewer.current_step == 2
    viewer.close()


def test_set_scalar_field_and_display_temperature() -> None:
    viewer = _viewer()
    viewer.set_scalar_field("temperature")
    viewer.display()
    assert viewer._current_grid is not None
    assert "temperature" in viewer._current_grid.point_data
    viewer.close()


def test_set_scalar_field_stress() -> None:
    viewer = _viewer()
    viewer.set_scalar_field("von_mises_stress")
    viewer.display()
    assert "von_mises_stress" in viewer._current_grid.cell_data
    viewer.close()


def test_set_scalar_field_vector_without_component_raises() -> None:
    viewer = _viewer()
    viewer.set_scalar_field("displacement")
    with pytest.raises(ValidationError):
        viewer.display()
    viewer.close()


def test_set_scalar_field_vector_with_component() -> None:
    viewer = _viewer()
    viewer.set_scalar_field("displacement", component=0)
    viewer.display()
    assert "displacement[0]" in viewer._current_grid.point_data
    viewer.close()


def test_set_scalar_field_missing_field_raises() -> None:
    viewer = _viewer()
    viewer.set_scalar_field("does_not_exist")
    with pytest.raises(ValidationError):
        viewer.display()
    viewer.close()


def test_set_scalar_field_none_clears_selection() -> None:
    viewer = _viewer()
    viewer.set_scalar_field("temperature")
    viewer.set_scalar_field(None)
    viewer.display()
    viewer.close()


def test_set_vector_field_heat_flux() -> None:
    viewer = _viewer()
    viewer.set_vector_field("heat_flux")
    viewer.display()
    assert "fea_vectors" in viewer._actors
    viewer.close()


def test_set_vector_field_displacement() -> None:
    viewer = _viewer()
    viewer.set_vector_field("displacement")
    viewer.display()
    assert "fea_vectors" in viewer._actors
    viewer.close()


def test_set_vector_field_missing_raises() -> None:
    viewer = _viewer()
    viewer.set_vector_field("does_not_exist")
    with pytest.raises(ValidationError):
        viewer.display()
    viewer.close()


def test_set_vector_field_none_removes_glyphs() -> None:
    viewer = _viewer()
    viewer.set_vector_field("heat_flux")
    viewer.display()
    viewer.set_vector_field(None)
    viewer.display()
    assert "fea_vectors" not in viewer._actors
    viewer.close()


def test_show_edges_toggle() -> None:
    viewer = _viewer()
    viewer.show_edges(False)
    assert viewer._show_edges is False
    viewer.show_edges(True)
    assert viewer._show_edges is True
    viewer.close()


def test_show_colorbar_toggle() -> None:
    viewer = _viewer()
    viewer.show_colorbar(False)
    assert viewer._show_colorbar is False
    viewer.close()


def test_show_undeformed_is_default() -> None:
    viewer = _viewer()
    viewer.display()
    original = np.array(viewer._current_grid.points)
    assert original == pytest.approx(np.array(list(_HEX8_COORDS.values())))
    viewer.close()


def test_show_deformed_with_zero_displacement_matches_original() -> None:
    topology = _topology()
    step = ResultStep(
        index=0, time=0.0, nodal_fields={"displacement": _zero_displacement()}
    )
    result = SimulationResult(topology=topology, steps=(step,))
    viewer = FEAViewer(result, ViewerConfig(off_screen=True))
    viewer.show_deformed()
    viewer.display()
    assert viewer._current_grid.points == pytest.approx(
        np.array(list(_HEX8_COORDS.values()))
    )
    viewer.close()


def test_show_deformed_with_known_displacement_and_scale() -> None:
    topology = _topology()
    step = ResultStep(
        index=0, time=0.0, nodal_fields={"displacement": _known_displacement()}
    )
    result = SimulationResult(topology=topology, steps=(step,))
    viewer = FEAViewer(result, ViewerConfig(off_screen=True))
    viewer.show_deformed()
    viewer.set_deformation_scale(2.0)
    viewer.display()

    expected = np.array(
        [np.array(coord) + 2.0 * np.array([0.01, 0.0, 0.0]) for coord in _HEX8_COORDS.values()]
    )
    assert viewer._current_grid.points == pytest.approx(expected)
    viewer.close()


def test_show_deformed_missing_field_raises() -> None:
    viewer = _viewer()
    viewer.show_deformed(displacement_field="does_not_exist")
    with pytest.raises(ValidationError):
        viewer.display()
    viewer.close()


def test_show_undeformed_after_deformed_reverts() -> None:
    viewer = _viewer()
    viewer.show_deformed()
    viewer.show_undeformed()
    viewer.display()
    assert viewer._current_grid.points == pytest.approx(
        np.array(list(_HEX8_COORDS.values()))
    )
    viewer.close()


def test_set_deformation_scale_non_finite_raises() -> None:
    viewer = _viewer()
    with pytest.raises(ValidationError):
        viewer.set_deformation_scale(float("nan"))
    with pytest.raises(ValidationError):
        viewer.set_deformation_scale(float("inf"))
    viewer.close()


def test_set_deformation_scale_negative_allowed() -> None:
    viewer = _viewer()
    viewer.set_deformation_scale(-5.0)
    assert viewer._deformation_scale == pytest.approx(-5.0)
    viewer.close()


def test_set_step_valid_and_current_step() -> None:
    viewer = _viewer()
    viewer.set_step(0)
    assert viewer.current_step == 0
    viewer.set_step(-1)
    assert viewer.current_step == 2
    viewer.close()


def test_set_step_out_of_range_raises() -> None:
    viewer = _viewer()
    with pytest.raises(ValidationError):
        viewer.set_step(10)
    with pytest.raises(ValidationError):
        viewer.set_step(-10)
    viewer.close()


def test_step_forward_and_backward_wrap() -> None:
    viewer = _viewer()
    viewer.set_step(2)
    viewer.step_forward()
    assert viewer.current_step == 0
    viewer.step_backward()
    assert viewer.current_step == 2
    viewer.close()


def test_add_and_remove_mesh() -> None:
    viewer = _viewer()
    viewer.set_scalar_field("temperature")
    viewer.display()
    assert "fea_mesh" in viewer._actors
    viewer.remove_mesh("fea_mesh")
    assert "fea_mesh" not in viewer._actors
    viewer.close()


def test_set_camera_named_presets() -> None:
    viewer = _viewer()
    for view in ("isometric", "xy", "xz", "yz"):
        viewer.set_camera(view)
    viewer.close()


def test_set_camera_invalid_name_raises() -> None:
    viewer = _viewer()
    with pytest.raises(ValidationError):
        viewer.set_camera("not_a_view")
    viewer.close()


def test_add_labels() -> None:
    viewer = _viewer()
    viewer.add_labels([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)], ["a", "b"])
    viewer.close()


def test_clip() -> None:
    viewer = _viewer()
    viewer.set_scalar_field("temperature")
    viewer.display()
    viewer.clip(normal=(1.0, 0.0, 0.0))
    viewer.close()


def test_slice() -> None:
    viewer = _viewer()
    viewer.set_scalar_field("temperature")
    viewer.display()
    viewer.slice(normal=(1.0, 0.0, 0.0))
    viewer.close()


def test_threshold() -> None:
    viewer = _viewer()
    viewer.set_scalar_field("temperature")
    viewer.display()
    viewer.threshold("temperature", (0.0, 1000.0))
    viewer.close()


def test_clip_without_prior_display_builds_grid() -> None:
    viewer = _viewer()
    viewer.clip(normal=(1.0, 0.0, 0.0))
    assert viewer._current_grid is not None
    viewer.close()


def test_probe_nearest_node() -> None:
    viewer = _viewer()
    node_id, value = viewer.probe_nearest_node((0.0, 0.0, 0.0), "temperature")
    assert node_id == 1
    assert value == pytest.approx(320.0)
    viewer.close()


def test_probe_element() -> None:
    viewer = _viewer()
    value = viewer.probe_element(1, "von_mises_stress")
    assert value == pytest.approx(1.0e6)
    viewer.close()


def test_animate_updates_step_without_recording() -> None:
    viewer = _viewer()
    viewer.animate(field_name="temperature")
    assert viewer.current_step == 2
    viewer.close()


def test_animate_with_explicit_steps() -> None:
    viewer = _viewer()
    viewer.animate(field_name="temperature", steps=[1, 0])
    assert viewer.current_step == 0
    viewer.close()


def test_animate_invalid_fps_raises() -> None:
    viewer = _viewer()
    with pytest.raises(ValidationError):
        viewer.animate(fps=0.0)
    with pytest.raises(ValidationError):
        viewer.animate(fps=-1.0)
    viewer.close()


def test_animate_records_gif(tmp_path) -> None:
    viewer = _viewer()
    gif_path = str(tmp_path / "animation.gif")
    viewer.animate(field_name="temperature", filename=gif_path, fps=2.0)
    assert os.path.exists(gif_path)
    viewer.close()


def test_screenshot_export(tmp_path) -> None:
    viewer = _viewer()
    viewer.set_scalar_field("temperature")
    viewer.display()
    image_path = str(tmp_path / "shot.png")
    viewer.screenshot(image_path)
    assert os.path.exists(image_path)
    viewer.close()


def test_export_mesh(tmp_path) -> None:
    viewer = _viewer()
    viewer.set_scalar_field("temperature")
    viewer.display()
    mesh_path = str(tmp_path / "mesh.vtu")
    viewer.export_mesh(mesh_path)
    assert os.path.exists(mesh_path)
    viewer.close()


def test_show_and_close_off_screen() -> None:
    viewer = _viewer()
    viewer.set_scalar_field("temperature")
    viewer.display()
    viewer.show()
    viewer.close()
