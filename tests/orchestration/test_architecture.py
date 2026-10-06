"""Architecture tests for the Version 34 orchestration layer.

Checks the layering boundaries the spec requires: the execution layer
has no GUI dependency, the simulation core does not depend on parallel
execution, and neither optimization nor uncertainty analysis manages
worker processes directly -- every one of them goes through
:mod:`femtoolkit.orchestration`'s centralized execution responsibility
instead of importing :mod:`concurrent.futures`/:mod:`multiprocessing`
itself.
"""

from __future__ import annotations

import ast
from pathlib import Path

import femtoolkit

SRC_ROOT = Path(femtoolkit.__file__).resolve().parent


def _imported_module_names(file_path: Path) -> set[str]:
    tree = ast.parse(file_path.read_text(), filename=str(file_path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def _all_py_files(package_dir: Path) -> list[Path]:
    return sorted(package_dir.rglob("*.py"))


def test_orchestration_package_has_no_gui_dependency() -> None:
    orchestration_dir = SRC_ROOT / "orchestration"
    for file_path in _all_py_files(orchestration_dir):
        imports = _imported_module_names(file_path)
        gui_imports = {name for name in imports if name.startswith("femtoolkit.gui")}
        assert not gui_imports, f"{file_path} imports GUI modules: {gui_imports}"


def test_run_manager_does_not_depend_on_orchestration() -> None:
    # The simulation core (femtoolkit.runs) must stay usable with zero
    # knowledge of the parallel-execution layer built on top of it --
    # femtoolkit.orchestration.simulation depends on femtoolkit.runs,
    # never the other way around.
    file_path = SRC_ROOT / "runs" / "manager.py"
    imports = _imported_module_names(file_path)
    orchestration_imports = {
        name for name in imports if name.startswith("femtoolkit.orchestration")
    }
    assert not orchestration_imports


def test_runs_package_does_not_depend_on_orchestration() -> None:
    runs_dir = SRC_ROOT / "runs"
    for file_path in _all_py_files(runs_dir):
        imports = _imported_module_names(file_path)
        orchestration_imports = {
            name for name in imports if name.startswith("femtoolkit.orchestration")
        }
        assert not orchestration_imports, (
            f"{file_path} depends on orchestration: {orchestration_imports}"
        )


def test_optimization_algorithms_do_not_manage_worker_processes_directly() -> None:
    # Population-based algorithms batch-evaluate through
    # femtoolkit.optimization.batch -> femtoolkit.orchestration; they
    # must never import concurrent.futures/multiprocessing themselves.
    algorithms_dir = SRC_ROOT / "optimization" / "algorithms"
    forbidden = {"concurrent.futures", "multiprocessing"}
    for file_path in _all_py_files(algorithms_dir):
        imports = _imported_module_names(file_path)
        hits = imports & forbidden
        assert not hits, f"{file_path} manages worker processes directly: {hits}"


def test_optimization_batch_is_the_only_worker_process_entry_point_in_optimization() -> None:
    forbidden = {"concurrent.futures", "multiprocessing"}
    optimization_dir = SRC_ROOT / "optimization"
    offenders = []
    for file_path in _all_py_files(optimization_dir):
        if file_path.name == "batch.py":
            continue
        imports = _imported_module_names(file_path)
        if imports & forbidden:
            offenders.append(file_path)
    assert not offenders, f"Unexpected direct process-pool usage in: {offenders}"


def test_uncertainty_package_does_not_manage_process_pools_directly() -> None:
    forbidden = {"concurrent.futures", "multiprocessing"}
    uncertainty_dir = SRC_ROOT / "uncertainty"
    for file_path in _all_py_files(uncertainty_dir):
        imports = _imported_module_names(file_path)
        hits = imports & forbidden
        assert not hits, f"{file_path} manages process pools directly: {hits}"


def test_process_pool_usage_is_centralized_in_orchestration_backends() -> None:
    # concurrent.futures.ProcessPoolExecutor is only ever constructed in
    # one place in the whole toolkit's independent-task layer.
    forbidden = {"concurrent.futures"}
    offenders = []
    for package_name in ("studies", "optimization", "uncertainty"):
        package_dir = SRC_ROOT / package_name
        for file_path in _all_py_files(package_dir):
            imports = _imported_module_names(file_path)
            if imports & forbidden:
                offenders.append(file_path)
    assert not offenders, f"concurrent.futures used outside orchestration in: {offenders}"
