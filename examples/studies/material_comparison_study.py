"""Example: comparing Steel/Aluminum/Titanium under the same loading (Version 30).

**Procedure.** Builds one base cantilever beam project and three
explicit :class:`~femtoolkit.studies.scenarios.Scenario` overrides, one
per material preset from the existing Version 24
:mod:`femtoolkit.application.materials_catalog` (no new material data
is introduced here), runs all three under the identical geometry,
boundary conditions, and tip load, and reports the resulting
displacement, stress, density, and execution time side by side.

**This example deliberately does not rank the materials or declare a
"winner."** Which material is "best" depends on engineering
requirements this toolkit has no way to know -- allowable stress,
weight budget, cost, corrosion resistance, fatigue life, manufacturing
constraints -- none of which are inputs here. The toolkit's job is to
report engineering quantities accurately; interpreting them against a
real design's requirements is the engineer's job, not this script's.
"""

from __future__ import annotations

from femtoolkit.application.materials_catalog import get_material_preset
from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.studies import (
    Scenario,
    SimulationStudy,
    StudyRunner,
    build_study_report,
    get_extractor,
    save_study_report,
)

_MATERIAL_PRESETS = ["structural_steel", "aluminum_6061", "titanium_ti6al4v"]


def main() -> None:
    """Run the material comparison study and print/save its results."""
    print("Finite Element Toolkit")
    print("Version 30 -- Material Comparison Study")
    print("=" * 45)

    base_project = Project(name="Material Comparison Study", analysis_type="linear_static")
    base_project.material = get_material_preset("structural_steel")
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

    scenarios = []
    for preset_key in _MATERIAL_PRESETS:
        preset = get_material_preset(preset_key)
        scenarios.append(
            Scenario(
                scenario_id=preset_key,
                name=preset.name,
                description=f"Same geometry and loading, material set to {preset.name}.",
                parameter_overrides={
                    "material.name": preset.name,
                    "material.youngs_modulus": preset.youngs_modulus,
                    "material.poisson_ratio": preset.poisson_ratio,
                    "material.density": preset.density,
                },
            )
        )

    study = SimulationStudy(
        study_id="material-comparison-study",
        name="Material Comparison Study",
        base_project=base_project,
        scenarios=scenarios,
    )
    result = StudyRunner().run(study)
    print(f"\n{len(result.successful_runs)}/{len(result.runs)} runs succeeded.")

    displacement = get_extractor("maximum_displacement")
    stress = get_extractor("maximum_von_mises_stress")

    print("\nMaterial | Density (kg/m^3) | Max displacement (m) | Max von Mises stress (Pa)")
    for run in result.successful_runs:
        material = run.configuration_snapshot.material
        print(
            f"{material.name:<20} | {material.density:>16.1f} | {displacement(run):>21.6e} | "
            f"{stress(run):>25.6e}"
        )

    comparison = result.compare(displacement, "Maximum displacement")
    print(f"\nDisplacement comparison (baseline: {comparison.baseline_run_id[:8]}):")
    for entry in comparison.entries:
        print(f"  run {entry.run_id_2[:8]}: {entry.percentage_change:+.1f}% vs. baseline")

    print(
        "\nNo 'best material' is declared -- the table above reports engineering "
        "quantities only. Which material is appropriate depends on requirements "
        "(allowable stress, weight, cost, corrosion resistance, ...) that this study "
        "does not know and does not assume."
    )

    report = build_study_report(
        title="Material Comparison Study Report",
        study_summary=(
            "Compares Structural Steel, Aluminum 6061-T6, and Titanium Ti-6Al-4V under "
            "identical geometry, boundary conditions, and tip load. Reports engineering "
            "quantities only -- no material ranking or recommendation is made."
        ),
        base_model_description=(
            "2.0 m x 0.4 m x 0.02 m cantilever beam, fixed (X, Y) at the left edge, "
            "-1000 N tip load per node on the right edge."
        ),
        result=result,
        comparisons=[comparison],
        conclusions=(
            "Displacement and stress differ across materials as expected from their "
            "differing Young's moduli; density differs independently of stiffness. "
            "Selecting a material for a real design requires engineering requirements "
            "(allowable stress, weight budget, cost, environment) outside this study's "
            "scope -- no material is identified as 'best' here."
        ),
    )
    save_study_report(report, "examples/studies/material_comparison_study_report.md", "markdown")
    print("\nSaved report: examples/studies/material_comparison_study_report.md")


if __name__ == "__main__":
    main()
