"""Example: empirical limit-exceedance estimation (Version 31).

**Procedure.** Defines an engineering displacement limit (3 mm) on a
cantilever beam under an uncertain tip load, then estimates
``P(u > 3 mm)`` as the empirical fraction of Monte Carlo samples whose
tip displacement exceeds that limit. This is explicitly reported as an
*empirical Monte Carlo estimate* from a finite sample -- not a rigorous
reliability index (no FORM/SORM or importance sampling is implemented
in this version; see ``docs/uncertainty.md``).
"""

from __future__ import annotations

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.studies.extractors import get_extractor
from femtoolkit.uncertainty import (
    MonteCarloConfig,
    MonteCarloRunner,
    UncertainParameter,
    UniformDistribution,
    exceedance_probability,
    plot_output_histogram,
)

DISPLACEMENT_LIMIT_M = 0.003


def main() -> None:
    """Run the limit-exceedance study and print/save its results."""
    print("Finite Element Toolkit")
    print("Version 31 -- Limit Exceedance Study")
    print("=" * 45)

    base_project = Project(name="Limit Exceedance Study", analysis_type="linear_static")
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
    base_project.loads = [LoadConfig(region="right", dof="Y", magnitude=-4000.0)]

    load_magnitude = UncertainParameter(
        path="loads.0.magnitude",
        label="Tip load magnitude",
        distribution=UniformDistribution(low=-6000.0, high=-2000.0),
        units="N",
        description="Service tip load, deliberately wide enough that some samples exceed "
        "the displacement limit.",
    )

    seed = 2024
    n_samples = 500
    config = MonteCarloConfig(
        study_id="limit-exceedance-study",
        name="Limit Exceedance Study",
        base_project=base_project,
        parameters=[load_magnitude],
        output_quantities=["maximum_displacement"],
        n_samples=n_samples,
        seed=seed,
        method="random",
    )
    result = MonteCarloRunner().run(config)
    print(f"\n{result.n_successful}/{config.n_samples} runs succeeded.")

    displacement = get_extractor("maximum_displacement")
    values = result.output_values(displacement)

    exceedance = exceedance_probability(
        values, threshold=DISPLACEMENT_LIMIT_M, quantity_label="Maximum displacement"
    )
    print(f"\nDisplacement limit: {DISPLACEMENT_LIMIT_M * 1000:.1f} mm")
    print(f"Sampling method: {config.method}, seed: {seed}")
    print(f"Samples evaluated: {exceedance.n_samples}")
    print(f"Samples exceeding the limit: {exceedance.n_exceeding}")
    print(f"Estimated exceedance frequency: {exceedance.exceedance_frequency:.2%}")
    print(
        "\nThis is an empirical Monte Carlo estimate from a finite sample -- not a "
        "rigorous reliability index. Its own precision is limited by the sample count "
        "above; a materially different result should be expected from a different seed "
        "or a larger sample, especially for a rare event."
    )

    figure = plot_output_histogram(values, "Maximum displacement", units="m")
    axes = figure.axes[0]
    axes.axvline(DISPLACEMENT_LIMIT_M, color="red", linestyle="--", label="Displacement limit")
    axes.legend()
    figure.savefig("examples/uncertainty/limit_exceedance_displacement.png")
    print("\nSaved plot: examples/uncertainty/limit_exceedance_displacement.png")


if __name__ == "__main__":
    main()
