"""Example: multiple uncertain inputs together (Version 31).

**Procedure.** Combines three simultaneously uncertain inputs on the
same cantilever beam -- tip load (normal), Young's modulus (normal),
and thickness (uniform) -- and analyzes their *combined* effect on tip
displacement, using a modest sample count (as recommended for a
multi-parameter study, spec section 37) and Latin Hypercube sampling
for better input-space coverage per sample. Reports which input the
combined displacement uncertainty correlates with most strongly --
a descriptive ranking of statistical association, not a claim about
physical importance.
"""

from __future__ import annotations

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.studies.extractors import get_extractor
from femtoolkit.uncertainty import (
    MonteCarloConfig,
    MonteCarloRunner,
    NormalDistribution,
    UncertainParameter,
    UniformDistribution,
    compute_output_statistics,
    correlation_summary,
    plot_output_histogram,
)


def main() -> None:
    """Run the multi-parameter uncertainty study and print/save its results."""
    print("Finite Element Toolkit")
    print("Version 31 -- Multi-Parameter Uncertainty Study")
    print("=" * 45)

    base_project = Project(name="Multi-Parameter Uncertainty Study", analysis_type="linear_static")
    base_project.material.youngs_modulus = 200e9
    base_project.material.poisson_ratio = 0.3
    base_project.mesh.width = 2.0
    base_project.mesh.height = 0.4
    base_project.mesh.nx = 16
    base_project.mesh.ny = 4
    base_project.mesh.thickness = 0.02
    base_project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    base_project.loads = [LoadConfig(region="right", dof="Y", magnitude=-2000.0)]

    parameters = [
        UncertainParameter(
            path="loads.0.magnitude",
            label="Tip load",
            distribution=NormalDistribution(mean_value=-2000.0, std_value=200.0),
            units="N",
            physical_upper_bound=0.0,
        ),
        UncertainParameter(
            path="material.youngs_modulus",
            label="Young's Modulus",
            distribution=NormalDistribution(mean_value=200e9, std_value=5e9),
            units="Pa",
            physical_lower_bound=0.0,
        ),
        UncertainParameter(
            path="mesh.thickness",
            label="Thickness",
            distribution=UniformDistribution(low=0.018, high=0.022),
            units="m",
            physical_lower_bound=0.0,
        ),
    ]

    config = MonteCarloConfig(
        study_id="multi-parameter-uncertainty-study",
        name="Multi-Parameter Uncertainty Study",
        base_project=base_project,
        parameters=parameters,
        output_quantities=["maximum_displacement"],
        n_samples=80,
        seed=11,
        method="latin_hypercube",
    )
    result = MonteCarloRunner().run(config)
    print(f"\n{result.n_successful}/{config.n_samples} runs succeeded.")

    displacement = get_extractor("maximum_displacement")
    values = result.output_values(displacement)
    stats = compute_output_statistics(values, "Maximum displacement", units="m")
    print(
        f"\nCombined effect on maximum displacement (m): "
        f"mean={stats.mean:.6e}  std={stats.std:.6e}"
    )

    figure = plot_output_histogram(values, "Maximum displacement", units="m")
    figure.savefig("examples/uncertainty/multi_parameter_uncertainty_displacement.png")

    print("\nCorrelation of each input with maximum displacement (descriptive only):")
    pairs = {}
    for parameter in parameters:
        x, y = result.successful_pairs(parameter.path, displacement)
        pairs[parameter.path] = (parameter.label, x, y)
    summary = correlation_summary("Maximum displacement", pairs)
    for entry in summary:
        print(f"  {entry.parameter_label:<16} Pearson r={entry.pearson_r:+.4f}")
    print(
        "\nThe largest-magnitude correlation above reflects this sample's linear "
        "association only -- it is not, on its own, an engineering statement about "
        "which parameter matters most for a real design."
    )


if __name__ == "__main__":
    main()
