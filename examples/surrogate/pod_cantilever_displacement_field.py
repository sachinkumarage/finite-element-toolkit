"""Example: Proper Orthogonal Decomposition of cantilever displacement fields (Version 35).

**Procedure.** Runs the same cantilever mesh under several different
load/thickness combinations, collects the *full* nodal displacement
field from each completed run (not just a scalar maximum displacement)
with :func:`~femtoolkit.surrogate.rom.snapshots.collect_field_snapshots_from_runs`,
builds a POD reduced basis from a training subset with
:class:`~femtoolkit.surrogate.rom.pod.PODModel`, and reconstructs two
held-out cases the basis never saw, reporting the retained modes,
captured energy, and reconstruction error for each (spec section 21:
training reconstruction accuracy is never assumed to generalize).
"""

from __future__ import annotations

import numpy as np

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.runs.manager import SimulationRunManager
from femtoolkit.studies.scenarios import Scenario, apply_scenario
from femtoolkit.surrogate.rom.pod import PODModel
from femtoolkit.surrogate.rom.snapshots import (
    build_snapshot_matrix,
    collect_field_snapshots_from_runs,
)


def build_base_project() -> Project:
    project = Project(name="POD Cantilever", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.material.density = 7850.0
    project.mesh.width = 2.0
    project.mesh.height = 0.4
    project.mesh.nx = 10
    project.mesh.ny = 3
    project.mesh.thickness = 0.01
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-3000.0)]
    return project


def _extract_displacement_field(run) -> np.ndarray | None:
    if run.result is None or run.result.simulation is None:
        return None
    step = run.result.simulation.final_step
    topology = run.result.simulation.topology
    return np.concatenate(
        [step.nodal_value("displacement", node_id) for node_id in topology.node_ids]
    )


def main() -> None:
    base_project = build_base_project()
    design_points = [
        {"mesh.thickness": thickness, "loads.0.magnitude": load}
        for thickness in [0.008, 0.010, 0.012, 0.014]
        for load in [-2000.0, -4000.0]
    ]
    held_out_points = [{"mesh.thickness": 0.009, "loads.0.magnitude": -3000.0}]

    manager = SimulationRunManager()

    def run_points(points: list[dict[str, float]], prefix: str):
        runs = []
        for index, point in enumerate(points):
            scenario = Scenario(
                scenario_id=f"{prefix}-{index}",
                name=f"{prefix} {index}",
                parameter_overrides=dict(point),
            )
            project = apply_scenario(base_project, scenario)
            runs.append(manager.execute(project, scenario_id=scenario.scenario_id))
        return runs

    training_runs = run_points(design_points, "pod-train")
    held_out_runs = run_points(held_out_points, "pod-holdout")

    training_snapshots = collect_field_snapshots_from_runs(
        training_runs, design_points, _extract_displacement_field
    )
    held_out_snapshots = collect_field_snapshots_from_runs(
        held_out_runs, held_out_points, _extract_displacement_field
    )
    print(f"Collected {len(training_snapshots)} training field snapshots.")

    snapshot_matrix = build_snapshot_matrix(training_snapshots)
    pod = PODModel().fit(snapshot_matrix, energy_threshold=0.999)
    print(
        f"POD basis: {pod.selected_modes}/{pod.total_modes} modes retained, "
        f"captured energy={pod.captured_energy:.6f}, discarded energy={pod.discarded_energy:.2e}"
    )

    print("\nReconstruction error on held-out cases (never seen by the POD basis):")
    for snapshot in held_out_snapshots:
        reduced = pod.reduce(snapshot.field)
        reconstructed = pod.reconstruct(reduced)
        absolute_error = float(np.linalg.norm(snapshot.field - reconstructed))
        relative_error = absolute_error / (float(np.linalg.norm(snapshot.field)) + 1e-12)
        print(
            f"  design point {snapshot.design_point}: "
            f"absolute L2 error={absolute_error:.3e} m, relative error={relative_error:.3e}"
        )


if __name__ == "__main__":
    main()
