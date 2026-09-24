"""Example: a cantilever beam load parameter study (Version 30).

**Procedure.** Builds a base cantilever beam project (steel, fixed at
the left edge, a tip load on the right edge), sweeps the tip load over
1-5 kN with :class:`~femtoolkit.studies.parameter_sweep.ParameterDefinition`,
runs every resulting scenario sequentially through
:class:`~femtoolkit.studies.runner.StudyRunner` (which itself reuses the
existing Version 24 solver pipeline via
:class:`~femtoolkit.runs.manager.SimulationRunManager` -- no FEA
algorithm is duplicated here), compares tip displacement and maximum
stress across every run, and verifies from the *simulation results
themselves* -- never hard-coded -- that a linear-elastic beam's
displacement and stress scale linearly with the applied load (``u ~
P``, ``sigma ~ P``). Finally generates a thirteen-section study report
and a displacement-vs-load plot.
"""

from __future__ import annotations

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.studies import (
    ParameterDefinition,
    SimulationStudy,
    StudyRunner,
    build_study_report,
    get_extractor,
    plot_study_quantity,
    save_study_report,
)


def main() -> None:
    """Run the cantilever beam load study and print/save its results."""
    print("Finite Element Toolkit")
    print("Version 30 -- Cantilever Beam Load Study")
    print("=" * 45)

    base_project = Project(name="Cantilever Beam Load Study", analysis_type="linear_static")
    base_project.material.youngs_modulus = 200e9
    base_project.material.poisson_ratio = 0.3
    base_project.material.density = 7850.0
    base_project.mesh.width = 2.0
    base_project.mesh.height = 0.4
    base_project.mesh.nx = 20
    base_project.mesh.ny = 4
    base_project.mesh.thickness = 0.02
    base_project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    base_project.loads = [LoadConfig(region="right", dof="Y", magnitude=-1000.0)]

    tip_load = ParameterDefinition(
        path="loads.0.magnitude",
        label="Tip load magnitude (N)",
        values=[-1000.0, -2000.0, -3000.0, -4000.0, -5000.0],
    )
    study = SimulationStudy(
        study_id="cantilever-load-study",
        name="Cantilever Load Study",
        base_project=base_project,
        parameters=[tip_load],
    )

    result = StudyRunner().run(study)
    print(f"\n{len(result.successful_runs)}/{len(result.runs)} runs succeeded.")
    print(f"Verification summary: {result.verification_summary()}")

    displacement = get_extractor("maximum_displacement")
    stress = get_extractor("maximum_von_mises_stress")

    def reaction_force_y(run) -> float:
        """The total Y reaction force, reusing the Version 29 equilibrium check already
        computed automatically for this run -- no new force-summation logic here."""
        for component in run.result.equilibrium_check.components:
            if component.label == "Y":
                return component.reaction_total
        raise AssertionError("No Y-direction equilibrium component was found.")

    print(
        "\nPer-node load (N) | Tip displacement (m) | Max von Mises stress (Pa) | "
        "Total Y reaction (N)"
    )
    for run in result.successful_runs:
        load_value = run.configuration_snapshot.loads[0].magnitude
        print(
            f"{load_value:>17.1f} | {displacement(run):>21.6e} | {stress(run):>25.6e} | "
            f"{reaction_force_y(run):>20.4f}"
        )

    def applied_force_y(run) -> float:
        for component in run.result.equilibrium_check.components:
            if component.label == "Y":
                return component.applied_total
        raise AssertionError("No Y-direction equilibrium component was found.")

    print("\nVerifying global force equilibrium (reaction ~= -total applied load) per run:")
    for run in result.successful_runs:
        applied = applied_force_y(run)
        reaction = reaction_force_y(run)
        assert abs(reaction + applied) < 1e-6 * abs(applied), "Reaction force imbalance detected."
        print(f"  total applied={applied:.1f} N, reaction={reaction:.4f} N")

    displacement_comparison = result.compare(displacement, "Maximum displacement")
    stress_comparison = result.compare(stress, "Maximum von Mises stress")

    print("\nVerifying linear-elastic proportionality (u ~ P, sigma ~ P), from simulation:")
    base_load = abs(result.successful_runs[0].configuration_snapshot.loads[0].magnitude)
    for run, disp_entry, stress_entry in zip(
        result.successful_runs[1:], displacement_comparison.entries, stress_comparison.entries,
        strict=True,
    ):
        load_ratio = abs(run.configuration_snapshot.loads[0].magnitude) / base_load
        displacement_ratio = disp_entry.value2 / disp_entry.value1
        stress_ratio = stress_entry.value2 / stress_entry.value1
        print(
            f"  load ratio={load_ratio:.2f}  displacement ratio={displacement_ratio:.4f}  "
            f"stress ratio={stress_ratio:.4f}"
        )
        assert abs(displacement_ratio - load_ratio) / load_ratio < 1e-6, (
            "Displacement did not scale linearly with load."
        )
        assert abs(stress_ratio - load_ratio) / load_ratio < 1e-6, (
            "Stress did not scale linearly with load."
        )
    print("Linear-elastic proportionality confirmed from simulation results.")

    sensitivity = result.sensitivity(tip_load, displacement, "Maximum displacement")
    print("\nSensitivity of tip displacement to load magnitude (S = (dy/y)/(dp/p)):")
    for entry in sensitivity:
        print(f"  p1={entry.p1:.0f} -> p2={entry.p2:.0f}: S={entry.sensitivity:.4f}")

    figure = plot_study_quantity(
        result, tip_load, displacement, "Maximum tip displacement (m)"
    )
    figure.savefig("examples/studies/cantilever_load_study_displacement.png")
    print("\nSaved plot: examples/studies/cantilever_load_study_displacement.png")

    report = build_study_report(
        title="Cantilever Beam Load Study Report",
        study_summary=(
            "Investigates tip displacement, maximum stress, and reaction force as the "
            "applied tip load is swept from 1 kN to 5 kN, verifying linear-elastic "
            "proportionality computed directly from simulation results."
        ),
        base_model_description=(
            "2.0 m x 0.4 m x 0.02 m structural steel cantilever beam (E=200 GPa, v=0.3), "
            "fixed (X, Y) at the left edge."
        ),
        result=result,
        comparisons=[displacement_comparison, stress_comparison],
        sensitivities=[sensitivity],
        plot_paths=["examples/studies/cantilever_load_study_displacement.png"],
        conclusions=(
            "Tip displacement and maximum von Mises stress both scale linearly with the "
            "applied tip load, as expected for a linear-elastic beam within this load "
            "range -- confirmed to within numerical tolerance from the simulation results "
            "themselves, not assumed."
        ),
    )
    save_study_report(report, "examples/studies/cantilever_load_study_report.md", "markdown")
    print("Saved report: examples/studies/cantilever_load_study_report.md")


if __name__ == "__main__":
    main()
