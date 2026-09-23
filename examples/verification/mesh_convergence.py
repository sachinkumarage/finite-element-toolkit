"""Example: a mesh convergence study on a cantilever plate (Version 29).

**Engineering problem.** A cantilever plate under a transverse tip load,
modeled with Q4 elements -- unlike the beam-theory example, this 2D
continuum model has no simple closed-form reference, so *verification*
here means checking that the result **converges** as the mesh is
refined, not comparing to an exact value.

**Procedure.** :func:`~femtoolkit.verification.convergence.run_mesh_convergence_study`
re-solves the same plate at four mesh resolutions (coarse to fine),
tracking the maximum tip displacement, and reports the relative change
between consecutive levels -- not assumed to be monotonic (see the
module docstring of :mod:`femtoolkit.verification.convergence`).
"""

from __future__ import annotations

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh.generator import create_quad_mesh
from femtoolkit.verification.convergence import (
    MeshConvergenceLevel,
    MeshConvergenceSample,
    run_mesh_convergence_study,
)
from femtoolkit.verification.plots import plot_error_vs_mesh_size, plot_mesh_convergence

X = TranslationDOF.X
Y = TranslationDOF.Y

_WIDTH = 4.0
_HEIGHT = 0.4
_TIP_LOAD = 5000.0


def _cantilever_level(nx: int, ny: int, label: str) -> MeshConvergenceLevel:
    def run() -> MeshConvergenceSample:
        material = LinearElastic2D(
            youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress"
        )
        mesh = create_quad_mesh(
            width=_WIDTH, height=_HEIGHT, nx=nx, ny=ny, material=material, thickness=0.02
        )
        analysis = StaticLinearAnalysis(mesh)
        for node in mesh.nodes:
            if node.x == 0.0:
                analysis.add_boundary_condition(BoundaryCondition(node.id, X, 0.0))
                analysis.add_boundary_condition(BoundaryCondition(node.id, Y, 0.0))
        tip_node = next(n for n in mesh.nodes if n.x == _WIDTH and n.y == 0.0)
        analysis.add_load(NodalLoad(tip_node.id, Y, -_TIP_LOAD))
        result = analysis.solve()
        return MeshConvergenceSample(
            num_nodes=len(mesh.nodes),
            num_elements=len(mesh.elements),
            num_dofs=result.dof_map.total_dofs,
            result_value=result.displacement(tip_node.id, dof=Y),
        )

    return MeshConvergenceLevel(label=label, mesh_size=_WIDTH / nx, run=run)


def main() -> None:
    """Run a four-level mesh convergence study on a cantilever plate's tip displacement."""
    print("Finite Element Toolkit")
    print("Version 29 -- Mesh Convergence Study")
    print("=" * 40)

    levels = [
        _cantilever_level(4, 1, "Coarse"),
        _cantilever_level(10, 2, "Medium"),
        _cantilever_level(20, 4, "Fine"),
        _cantilever_level(40, 8, "Finer"),
    ]
    study = run_mesh_convergence_study("Tip displacement", levels, tolerance=1e-3)

    print(f"\n{'Level':<8}{'DOFs':>8}{'Value (m)':>16}{'Relative Change':>18}")
    for point in study.points:
        change = "-" if point.relative_change is None else f"{point.relative_change:.3e}"
        print(f"{point.label:<8}{point.num_dofs:>8}{point.result_value:>16.6e}{change:>18}")

    print(f"\nStudy status: {study.status.value.upper()}")

    figure_1 = plot_mesh_convergence(study)
    figure_2 = plot_error_vs_mesh_size(study)
    figure_1.savefig("examples/verification/mesh_convergence_result.png", dpi=100)
    figure_2.savefig("examples/verification/mesh_convergence_error.png", dpi=100)
    print("\nSaved mesh_convergence_result.png and mesh_convergence_error.png")


if __name__ == "__main__":
    main()
