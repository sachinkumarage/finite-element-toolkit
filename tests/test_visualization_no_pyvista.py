"""Confirms the visualization_3d package degrades gracefully without PyVista (spec section 26).

Simulates a missing PyVista installation by making ``import pyvista``
raise ``ModuleNotFoundError`` (via a ``builtins.__import__`` patch) and
re-importing every ``visualization_3d`` submodule fresh. This verifies,
for real, rather than by inspection, that:

* importing :mod:`femtoolkit.postprocessing.visualization_3d` (and its
  submodules) always succeeds, even without PyVista;
* actually *using* any of its functionality raises a clear
  :class:`ImportError`;
* the core FEA toolkit (materials, elements, solvers) is entirely
  unaffected.
"""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest

_VIZ_MODULES = [
    "femtoolkit.postprocessing.visualization_3d",
    "femtoolkit.postprocessing.visualization_3d.mesh_converter",
    "femtoolkit.postprocessing.visualization_3d.config",
    "femtoolkit.postprocessing.visualization_3d.viewer",
    "femtoolkit.postprocessing.visualization_3d.boundary_conditions",
]


@pytest.fixture
def without_pyvista(monkeypatch):
    """Simulate a missing pyvista: import raises ModuleNotFoundError, modules re-imported fresh."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyvista" or name.startswith("pyvista."):
            raise ModuleNotFoundError("No module named 'pyvista'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    removed = {}
    for name in list(sys.modules):
        if name == "pyvista" or name.startswith("pyvista.") or name in _VIZ_MODULES:
            removed[name] = sys.modules.pop(name)

    yield

    for name in list(sys.modules):
        if name == "pyvista" or name.startswith("pyvista.") or name in _VIZ_MODULES:
            sys.modules.pop(name, None)
    for name, module in removed.items():
        if name == "pyvista" or name.startswith("pyvista."):
            continue
        sys.modules[name] = module
    for name in _VIZ_MODULES:
        importlib.import_module(name)


def test_visualization_3d_imports_without_pyvista(without_pyvista) -> None:
    for name in _VIZ_MODULES:
        module = importlib.import_module(name)
        assert module is not None


def test_require_pyvista_raises_clear_import_error(without_pyvista) -> None:
    mesh_converter = importlib.import_module(
        "femtoolkit.postprocessing.visualization_3d.mesh_converter"
    )
    with pytest.raises(ImportError, match="pyvista"):
        mesh_converter.require_pyvista()


def test_fea_viewer_construction_raises_without_pyvista(without_pyvista) -> None:
    viewer_module = importlib.import_module("femtoolkit.postprocessing.visualization_3d.viewer")
    result_model = importlib.import_module("femtoolkit.postprocessing.result_model")

    topology = result_model.MeshTopology(
        node_ids=(1, 2, 3),
        node_coordinates={1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (0.0, 1.0, 0.0)},
        element_ids=(),
        element_connectivity={},
        element_types={},
    )
    step = result_model.ResultStep(index=0, time=0.0)
    result = result_model.SimulationResult(topology=topology, steps=(step,))

    with pytest.raises(ImportError):
        viewer_module.FEAViewer(result)


def test_mesh_conversion_raises_without_pyvista(without_pyvista) -> None:
    mesh_converter = importlib.import_module(
        "femtoolkit.postprocessing.visualization_3d.mesh_converter"
    )
    result_model = importlib.import_module("femtoolkit.postprocessing.result_model")

    topology = result_model.MeshTopology(
        node_ids=(1, 2, 3),
        node_coordinates={1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (0.0, 1.0, 0.0)},
        element_ids=(),
        element_connectivity={},
        element_types={},
    )
    with pytest.raises(ImportError):
        mesh_converter.mesh_topology_to_grid(topology)


def test_core_fea_unaffected_without_pyvista(without_pyvista) -> None:
    materials = importlib.import_module("femtoolkit.materials")
    elastic = materials.LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    assert elastic.youngs_modulus == pytest.approx(200e9)

    static_linear = importlib.import_module("femtoolkit.analysis.static_linear")
    assert static_linear is not None
