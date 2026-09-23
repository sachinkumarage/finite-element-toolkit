"""Tests for femtoolkit.verification.convergence (Version 29)."""

from __future__ import annotations

import pytest

from femtoolkit.verification.convergence import (
    MeshConvergenceLevel,
    MeshConvergenceSample,
    run_mesh_convergence_study,
)
from femtoolkit.verification.status import VerificationStatus


def _level(
    label: str, size: float, value: float, nodes: int = 10, elements: int = 5
) -> MeshConvergenceLevel:
    return MeshConvergenceLevel(
        label=label,
        mesh_size=size,
        run=lambda: MeshConvergenceSample(nodes, elements, nodes, value),
    )


def test_first_level_has_no_relative_change() -> None:
    study = run_mesh_convergence_study("Q", [_level("Coarse", 1.0, 10.0)])
    assert study.points[0].relative_change is None


def test_relative_change_computed_between_consecutive_levels() -> None:
    study = run_mesh_convergence_study(
        "Q", [_level("Coarse", 1.0, 10.0), _level("Fine", 0.5, 11.0)]
    )
    expected = abs(11.0 - 10.0) / 11.0
    assert study.points[1].relative_change == pytest.approx(expected)


def test_status_pass_when_converged_within_tolerance() -> None:
    study = run_mesh_convergence_study(
        "Q",
        [_level("Coarse", 1.0, 10.0), _level("Fine", 0.5, 10.0000001)],
        tolerance=1e-3,
    )
    assert study.status is VerificationStatus.PASS


def test_status_warning_when_not_converged() -> None:
    study = run_mesh_convergence_study(
        "Q", [_level("Coarse", 1.0, 10.0), _level("Fine", 0.5, 20.0)], tolerance=1e-3
    )
    assert study.status is VerificationStatus.WARNING


def test_status_not_available_with_fewer_than_two_levels() -> None:
    study = run_mesh_convergence_study("Q", [_level("Only", 1.0, 10.0)])
    assert study.status is VerificationStatus.NOT_AVAILABLE


def test_status_not_available_with_zero_levels() -> None:
    study = run_mesh_convergence_study("Q", [])
    assert study.status is VerificationStatus.NOT_AVAILABLE
    assert study.final_relative_change is None


def test_convergence_need_not_be_monotonic() -> None:
    """An oscillating-then-settling sequence must be recorded faithfully, not rejected."""
    study = run_mesh_convergence_study(
        "Q",
        [
            _level("Coarse", 1.0, 10.0),
            _level("Medium", 0.5, 12.0),
            _level("Fine", 0.25, 10.5),
            _level("Finer", 0.125, 10.5001),
        ],
        tolerance=1e-3,
    )
    assert len(study.points) == 4
    assert study.status is VerificationStatus.PASS


def test_points_record_mesh_statistics() -> None:
    study = run_mesh_convergence_study("Q", [_level("Coarse", 1.0, 10.0, nodes=20, elements=8)])
    assert study.points[0].num_nodes == 20
    assert study.points[0].num_elements == 8


def test_final_relative_change_property() -> None:
    study = run_mesh_convergence_study(
        "Q", [_level("Coarse", 1.0, 10.0), _level("Fine", 0.5, 11.0)]
    )
    assert study.final_relative_change == study.points[-1].relative_change


def test_real_cantilever_mesh_refinement_reduces_relative_change() -> None:
    """A real FEA cantilever, refined four times: relative change should
    shrink toward the analytical Euler-Bernoulli tip deflection."""
    from femtoolkit.analysis import (
        BoundaryCondition,
        NodalLoad,
        StaticLinearAnalysis,
        TranslationDOF,
    )
    from femtoolkit.materials import LinearElastic2D
    from femtoolkit.mesh.generator import create_quad_mesh

    x, y = TranslationDOF.X, TranslationDOF.Y

    def cantilever_level(nx: int, ny: int, label: str) -> MeshConvergenceLevel:
        def run() -> MeshConvergenceSample:
            material = LinearElastic2D(
                youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress"
            )
            mesh = create_quad_mesh(
                width=4.0, height=0.4, nx=nx, ny=ny, material=material, thickness=0.02
            )
            analysis = StaticLinearAnalysis(mesh)
            for node in mesh.nodes:
                if node.x == 0.0:
                    analysis.add_boundary_condition(BoundaryCondition(node.id, x, 0.0))
                    analysis.add_boundary_condition(BoundaryCondition(node.id, y, 0.0))
            tip_node = next(n for n in mesh.nodes if n.x == 4.0 and n.y == 0.0)
            analysis.add_load(NodalLoad(tip_node.id, y, -5000.0))
            result = analysis.solve()
            return MeshConvergenceSample(
                num_nodes=len(mesh.nodes),
                num_elements=len(mesh.elements),
                num_dofs=result.dof_map.total_dofs,
                result_value=result.displacement(tip_node.id, dof=y),
            )

        return MeshConvergenceLevel(label=label, mesh_size=4.0 / nx, run=run)

    levels = [
        cantilever_level(4, 1, "Coarse"),
        cantilever_level(10, 2, "Medium"),
        cantilever_level(20, 4, "Fine"),
    ]
    study = run_mesh_convergence_study("Tip displacement", levels, tolerance=1.0)

    changes = [point.relative_change for point in study.points if point.relative_change is not None]
    assert len(changes) == 2
    assert changes[1] < changes[0]
