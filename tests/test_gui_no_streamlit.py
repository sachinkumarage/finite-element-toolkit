"""Confirms the core toolkit and application layer work without Streamlit (Version 24, section 18).

Mirrors the Version 23 ``test_visualization_no_pyvista.py`` pattern:
patches ``builtins.__import__`` to simulate a missing Streamlit
installation, then verifies that :mod:`femtoolkit.application` (the
whole service layer) and :mod:`femtoolkit.gui`/:mod:`femtoolkit.gui.state`
(the Streamlit-free parts of the GUI package) still import and work
correctly, while only :mod:`femtoolkit.gui.app` and the page modules --
which actually need Streamlit widgets -- raise a clear
``ModuleNotFoundError``. This is the concrete proof behind spec section
18's requirement: "a user should still be able to run pytest without
requiring the GUI to be launched."
"""

from __future__ import annotations

import builtins
import contextlib
import importlib
import sys

import pytest

_STREAMLIT_FREE_MODULES = [
    "femtoolkit.application",
    "femtoolkit.gui",
    "femtoolkit.gui.state",
    "femtoolkit.gui.visualization",
]
_STREAMLIT_DEPENDENT_MODULES = [
    "femtoolkit.gui.app",
    "femtoolkit.gui.components",
]


@pytest.fixture
def without_streamlit(monkeypatch):
    """Simulate a missing streamlit: import raises ModuleNotFoundError."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "streamlit" or name.startswith("streamlit."):
            raise ModuleNotFoundError("No module named 'streamlit'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    tracked = _STREAMLIT_FREE_MODULES + _STREAMLIT_DEPENDENT_MODULES
    removed = {}
    for name in list(sys.modules):
        if name == "streamlit" or name.startswith("streamlit.") or name in tracked:
            removed[name] = sys.modules.pop(name)

    yield

    for name in list(sys.modules):
        if name == "streamlit" or name.startswith("streamlit.") or name in tracked:
            sys.modules.pop(name, None)
    for name, module in removed.items():
        if name == "streamlit" or name.startswith("streamlit."):
            continue
        sys.modules[name] = module
    for name in _STREAMLIT_FREE_MODULES + _STREAMLIT_DEPENDENT_MODULES:
        with contextlib.suppress(ModuleNotFoundError):
            importlib.import_module(name)


def test_application_layer_imports_without_streamlit(without_streamlit) -> None:
    for name in _STREAMLIT_FREE_MODULES:
        module = importlib.import_module(name)
        assert module is not None


def test_application_layer_fully_functional_without_streamlit(without_streamlit) -> None:
    application = importlib.import_module("femtoolkit.application")

    service = application.ProjectService()
    project = service.create_project("No Streamlit")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.boundary_conditions = [
        application.BoundaryConditionConfig(region="left", dof="X", value=0.0)
    ]
    project.loads = [application.LoadConfig(region="right", dof="X", magnitude=100.0)]

    result = application.SimulationService().run(project)
    assert result.succeeded


def test_gui_app_raises_clear_error_without_streamlit(without_streamlit) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("femtoolkit.gui.app")


def test_core_fea_unaffected_without_streamlit(without_streamlit) -> None:
    materials = importlib.import_module("femtoolkit.materials")
    elastic = materials.LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    assert elastic.youngs_modulus == pytest.approx(200e9)
