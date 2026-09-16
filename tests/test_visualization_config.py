"""Tests for femtoolkit.postprocessing.visualization_3d.config."""

import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.postprocessing.visualization_3d.config import CAMERA_VIEWS, ViewerConfig


def test_default_settings() -> None:
    config = ViewerConfig()

    assert config.window_title == "Finite Element Toolkit -- 3D Viewer"
    assert config.background_color == "white"
    assert config.show_edges is True
    assert config.colormap == "viridis"
    assert config.show_colorbar is True
    assert config.deformation_scale == pytest.approx(1.0)
    assert config.vector_scale == pytest.approx(1.0)
    assert config.window_size == (1024, 768)
    assert config.camera_view == "isometric"
    assert config.off_screen is False


def test_custom_settings() -> None:
    config = ViewerConfig(
        window_title="Custom Viewer",
        background_color="black",
        show_edges=False,
        colormap="plasma",
        show_colorbar=False,
        deformation_scale=50.0,
        vector_scale=2.5,
        window_size=(640, 480),
        camera_view="xy",
        off_screen=True,
    )

    assert config.window_title == "Custom Viewer"
    assert config.background_color == "black"
    assert config.show_edges is False
    assert config.colormap == "plasma"
    assert config.show_colorbar is False
    assert config.deformation_scale == pytest.approx(50.0)
    assert config.vector_scale == pytest.approx(2.5)
    assert config.window_size == (640, 480)
    assert config.camera_view == "xy"
    assert config.off_screen is True


def test_negative_deformation_scale_is_allowed() -> None:
    config = ViewerConfig(deformation_scale=-10.0)
    assert config.deformation_scale == pytest.approx(-10.0)


def test_non_finite_deformation_scale_raises() -> None:
    with pytest.raises(ValidationError):
        ViewerConfig(deformation_scale=float("inf"))
    with pytest.raises(ValidationError):
        ViewerConfig(deformation_scale=float("nan"))


def test_non_positive_vector_scale_raises() -> None:
    with pytest.raises(ValidationError):
        ViewerConfig(vector_scale=0.0)
    with pytest.raises(ValidationError):
        ViewerConfig(vector_scale=-1.0)


def test_non_finite_vector_scale_raises() -> None:
    with pytest.raises(ValidationError):
        ViewerConfig(vector_scale=float("inf"))
    with pytest.raises(ValidationError):
        ViewerConfig(vector_scale=float("nan"))


def test_non_positive_window_size_raises() -> None:
    with pytest.raises(ValidationError):
        ViewerConfig(window_size=(0, 768))
    with pytest.raises(ValidationError):
        ViewerConfig(window_size=(1024, 0))
    with pytest.raises(ValidationError):
        ViewerConfig(window_size=(-1, 768))


def test_invalid_camera_view_raises() -> None:
    with pytest.raises(ValidationError):
        ViewerConfig(camera_view="birds_eye")


def test_all_camera_views_accepted() -> None:
    for view in CAMERA_VIEWS:
        config = ViewerConfig(camera_view=view)
        assert config.camera_view == view


def test_config_is_frozen() -> None:
    config = ViewerConfig()
    with pytest.raises(Exception):  # noqa: B017 - dataclasses.FrozenInstanceError
        config.window_title = "changed"  # type: ignore[misc]
