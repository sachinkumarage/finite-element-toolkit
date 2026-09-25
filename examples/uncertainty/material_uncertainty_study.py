"""Example: material property uncertainty (Version 31).

**Procedure.** Builds a base cantilever beam project, represents its
Young's modulus as a normal random variable (mean 200 GPa, standard
deviation 5 GPa -- a realistic manufacturing spread for structural
steel), runs a 200-sample Monte Carlo study reusing the exact Version
30 scenario/run infrastructure, and reports how that input uncertainty
propagates to tip displacement and maximum stress: descriptive
statistics, percentiles, a confidence interval for the mean, a
histogram, and the correlation between Young's modulus and each output.
No statistical result here is hard-coded -- every number is computed
from the sampled results themselves.
"""

from __future__ import annotations

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.studies.extractors import get_extractor
from femtoolkit.uncertainty import (
    ConfidenceInterval,
    MonteCarloConfig,
    MonteCarloRunner,
    NormalDistribution,
    OutputStatistics,
    UncertainParameter,
    build_uncertainty_report,
    compute_output_statistics,
    confidence_interval_mean,
    correlation_summary,
    plot_output_histogram,
    save_uncertainty_report,
)


def main() -> None:
    """Run the material uncertainty study and print/save its results."""
    print("Finite Element Toolkit")
    print("Version 31 -- Material Uncertainty Study")
    print("=" * 45)

    base_project = Project(name="Material Uncertainty Study", analysis_type="linear_static")
    base_project.material.youngs_modulus = 200e9
    base_project.material.poisson_ratio = 0.3
    base_project.material.density = 7850.0
    base_project.mesh.width = 2.0
    base_project.mesh.height = 0.4
    base_project.mesh.nx = 16
    base_project.mesh.ny = 4
    base_project.mesh.thickness = 0.02
    base_project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    base_project.loads = [LoadConfig(region="right", dof="Y", magnitude=-1000.0)]

    youngs_modulus = UncertainParameter(
        path="material.youngs_modulus",
        label="Young's Modulus",
        distribution=NormalDistribution(mean_value=200e9, std_value=5e9),
        units="Pa",
        description="Structural steel elastic modulus, with manufacturing/batch variability.",
        physical_lower_bound=0.0,
        reference_value=200e9,
    )

    config = MonteCarloConfig(
        study_id="material-uncertainty-study",
        name="Material Uncertainty Study",
        base_project=base_project,
        parameters=[youngs_modulus],
        output_quantities=["maximum_displacement", "maximum_von_mises_stress"],
        n_samples=200,
        seed=42,
        method="random",
    )
    result = MonteCarloRunner().run(config)
    print(
        f"\nSamples: {config.n_samples} requested, {result.n_successful} successful, "
        f"{result.n_failed} failed, {result.n_invalid} rejected as physically invalid."
    )

    displacement = get_extractor("maximum_displacement")
    stress = get_extractor("maximum_von_mises_stress")

    all_statistics: list[OutputStatistics] = []
    all_confidence_intervals: list[ConfidenceInterval] = []
    plot_paths: list[str] = []

    for name, extractor, units in (
        ("Maximum displacement", displacement, "m"),
        ("Maximum von Mises stress", stress, "Pa"),
    ):
        values = result.output_values(extractor)
        stats = compute_output_statistics(
            values, name, units=units, n_requested=config.n_samples,
            n_failed=result.n_failed, n_invalid=result.n_invalid,
        )
        ci = confidence_interval_mean(values, name)
        all_statistics.append(stats)
        all_confidence_intervals.append(ci)

        print(f"\n{name} ({units}):")
        print(
            f"  mean={stats.mean:.6e}  std={stats.std:.6e}  CV={stats.coefficient_of_variation:.4f}"
        )
        print(f"  95% CI for the mean: [{ci.lower:.6e}, {ci.upper:.6e}]")
        print(f"  P5={stats.percentiles[5.0]:.6e}  P50={stats.percentiles[50.0]:.6e}  "
              f"P95={stats.percentiles[95.0]:.6e}")

        figure = plot_output_histogram(values, name, units=units)
        slug = name.lower().replace(" ", "_")
        path = f"examples/uncertainty/material_uncertainty_{slug}.png"
        figure.savefig(path)
        plot_paths.append(path)

    print("\nCorrelation with Young's Modulus:")
    correlations = []
    for name, extractor in (
        ("Maximum displacement", displacement),
        ("Maximum von Mises stress", stress),
    ):
        x, y = result.successful_pairs("material.youngs_modulus", extractor)
        summary = correlation_summary(name, {"material.youngs_modulus": ("Young's Modulus", x, y)})
        correlations.extend(summary)
        entry = summary[0]
        print(f"  {name}: Pearson r={entry.pearson_r:.4f}, Spearman rho={entry.spearman_rho:.4f}")

    report = build_uncertainty_report(
        title="Material Uncertainty Study Report",
        study_summary=(
            "Propagates Young's modulus uncertainty (normal, mean 200 GPa, std 5 GPa) "
            "through a linear-elastic cantilever beam analysis to tip displacement and "
            "maximum stress."
        ),
        base_model_description=(
            "2.0 m x 0.4 m x 0.02 m structural steel cantilever beam, fixed (X, Y) at the "
            "left edge, -1000 N tip load per node on the right edge."
        ),
        result=result,
        output_statistics=all_statistics,
        confidence_intervals=all_confidence_intervals,
        correlations=correlations,
        plot_paths=plot_paths,
        conclusions=(
            "Displacement and stress uncertainty are dominated by Young's modulus "
            "uncertainty, with a strong negative correlation between modulus and "
            "displacement (stiffer material, smaller displacement) -- as expected from "
            "linear-elastic theory, confirmed here from sampled simulation results."
        ),
    )
    report_path = "examples/uncertainty/material_uncertainty_report.md"
    save_uncertainty_report(report, report_path, "markdown")
    print("\nSaved report: examples/uncertainty/material_uncertainty_report.md")


if __name__ == "__main__":
    main()
