"""Example: applied load uncertainty (Version 31).

**Procedure.** Represents the tip load on a cantilever beam as a
uniform random variable, ``P ~ U(1000 N, 3000 N)`` -- a plausible
service-load range rather than one exact design value -- and analyzes
the resulting uncertainty in tip displacement, maximum stress, and the
total reaction force at the fixed support (reusing the Version 29
equilibrium check already computed automatically for every run, not a
new force-summation routine).
"""

from __future__ import annotations

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.studies.extractors import get_extractor
from femtoolkit.uncertainty import (
    MonteCarloConfig,
    MonteCarloRunner,
    UncertainParameter,
    UniformDistribution,
    compute_output_statistics,
    plot_output_histogram,
)


def _reaction_force_y(run) -> float:
    """The total Y reaction force, reusing the Version 29 equilibrium check already
    computed automatically for this run -- no new force-summation logic here."""
    for component in run.result.equilibrium_check.components:
        if component.label == "Y":
            return component.reaction_total
    raise AssertionError("No Y-direction equilibrium component was found.")


def main() -> None:
    """Run the load uncertainty study and print/save its results."""
    print("Finite Element Toolkit")
    print("Version 31 -- Load Uncertainty Study")
    print("=" * 45)

    base_project = Project(name="Load Uncertainty Study", analysis_type="linear_static")
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

    load_magnitude = UncertainParameter(
        path="loads.0.magnitude",
        label="Tip load magnitude",
        distribution=UniformDistribution(low=-3000.0, high=-1000.0),
        units="N",
        description="Service tip load, varying between 1 kN and 3 kN downward.",
    )

    config = MonteCarloConfig(
        study_id="load-uncertainty-study",
        name="Load Uncertainty Study",
        base_project=base_project,
        parameters=[load_magnitude],
        output_quantities=["maximum_displacement", "maximum_von_mises_stress"],
        n_samples=150,
        seed=7,
        method="latin_hypercube",
    )
    result = MonteCarloRunner().run(config)
    print(f"\n{result.n_successful}/{config.n_samples} runs succeeded.")

    displacement = get_extractor("maximum_displacement")
    stress = get_extractor("maximum_von_mises_stress")

    for name, extractor, units in (
        ("Maximum displacement", displacement, "m"),
        ("Maximum von Mises stress", stress, "Pa"),
    ):
        values = result.output_values(extractor)
        stats = compute_output_statistics(values, name, units=units)
        print(f"\n{name} ({units}): mean={stats.mean:.6e}  std={stats.std:.6e}")
        figure = plot_output_histogram(values, name, units=units)
        figure.savefig(f"examples/uncertainty/load_uncertainty_{name.split()[-1]}.png")

    reaction_values = [
        _reaction_force_y(run) for run in result.study_result.successful_runs
    ]
    reaction_stats = compute_output_statistics(reaction_values, "Total Y reaction force", units="N")
    print(
        f"\nTotal Y reaction force (N): mean={reaction_stats.mean:.4f}  "
        f"std={reaction_stats.std:.4f}"
    )

    print(
        "\nVerifying global force equilibrium (reaction ~= -total applied load) per run:"
    )
    max_imbalance = 0.0
    for run in result.study_result.successful_runs:
        applied = next(
            component.applied_total
            for component in run.result.equilibrium_check.components
            if component.label == "Y"
        )
        reaction = _reaction_force_y(run)
        imbalance = abs(reaction + applied)
        max_imbalance = max(max_imbalance, imbalance)
    print(f"  max |reaction + applied| across all runs: {max_imbalance:.3e} N")
    assert max_imbalance < 1e-6, "Reaction force imbalance detected across the study."
    print("Global force equilibrium confirmed across the full sampled load range.")


if __name__ == "__main__":
    main()
