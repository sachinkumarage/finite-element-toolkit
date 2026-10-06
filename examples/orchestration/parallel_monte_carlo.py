"""Example: parallel Monte Carlo uncertainty study (Version 34).

**Procedure.** Builds a cantilever beam with two uncertain inputs --
Young's modulus (normal, a manufacturing spread) and the applied tip
load (uniform, an operational range) -- and runs the *same*
:class:`~femtoolkit.uncertainty.monte_carlo.MonteCarloConfig` three
times: serially, and in parallel with two different worker counts.

**What this example actually demonstrates.** The spec's hard
reproducibility requirement for Version 34 is that *changing the number
of workers must not change the generated uncertainty samples* --
sampling happens entirely before any task is built or executed (see
:mod:`femtoolkit.orchestration.random_state`'s module docstring), so
this is a structural guarantee, not a coincidence. This example proves
it empirically: it compares the three runs' raw sample arrays
element-by-element (not just their summary statistics, which could
coincidentally match even if the underlying samples differed) and
confirms they are exactly identical, then compares mean, standard
deviation, percentiles, and sample count computed from each run's
outputs -- all identical too, since identical inputs through an
identical (if differently-scheduled) pipeline must produce identical
outputs.
"""

from __future__ import annotations

import time

import numpy as np

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.orchestration.config import OrchestrationConfig
from femtoolkit.studies.extractors import get_extractor
from femtoolkit.uncertainty import (
    MonteCarloConfig,
    MonteCarloRunner,
    NormalDistribution,
    UncertainParameter,
    UniformDistribution,
    compute_output_statistics,
)


def build_base_project() -> Project:
    project = Project(name="Parallel Monte Carlo Study", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.material.density = 7850.0
    project.mesh.width = 2.0
    project.mesh.height = 0.4
    project.mesh.nx = 24
    project.mesh.ny = 6
    project.mesh.thickness = 0.02
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-3000.0)]
    return project


def main() -> None:
    """Run the same Monte Carlo study serially and in parallel, and compare them."""
    print("Finite Element Toolkit")
    print("Version 34 -- Parallel Monte Carlo Study")
    print("=" * 45)

    youngs_modulus = UncertainParameter(
        path="material.youngs_modulus", label="Young's Modulus",
        distribution=NormalDistribution(mean_value=200e9, std_value=5e9),
        units="Pa", physical_lower_bound=0.0,
    )
    tip_load = UncertainParameter(
        path="loads.0.magnitude", label="Tip load",
        distribution=UniformDistribution(low=-4000.0, high=-2000.0), units="N",
    )

    config = MonteCarloConfig(
        study_id="parallel-mc-study", name="Parallel Monte Carlo Study",
        base_project=build_base_project(), parameters=[youngs_modulus, tip_load],
        output_quantities=["maximum_displacement"], n_samples=60, seed=2024,
    )
    extractor = get_extractor("maximum_displacement")

    print(f"\nRunning {config.n_samples} samples serially...")
    start = time.perf_counter()
    serial_result = MonteCarloRunner().run(config)
    serial_elapsed = time.perf_counter() - start
    print(f"  {serial_result.n_successful}/{config.n_samples} succeeded in "
          f"{serial_elapsed:.2f}s.")

    print(f"\nRunning the same {config.n_samples} samples in parallel (2 workers)...")
    start = time.perf_counter()
    parallel_2w = MonteCarloRunner().run(
        config, orchestration_config=OrchestrationConfig(execution_mode="parallel", max_workers=2)
    )
    parallel_2w_elapsed = time.perf_counter() - start
    print(f"  {parallel_2w.n_successful}/{config.n_samples} succeeded in "
          f"{parallel_2w_elapsed:.2f}s.")

    print(f"\nRunning the same {config.n_samples} samples in parallel (4 workers)...")
    start = time.perf_counter()
    parallel_4w = MonteCarloRunner().run(
        config, orchestration_config=OrchestrationConfig(execution_mode="parallel", max_workers=4)
    )
    parallel_4w_elapsed = time.perf_counter() - start
    print(f"  {parallel_4w.n_successful}/{config.n_samples} succeeded in "
          f"{parallel_4w_elapsed:.2f}s.")

    print("\nVerifying the raw sample arrays are bit-identical regardless of worker count:")
    samples_match_2w = np.array_equal(
        serial_result.sample_set.values, parallel_2w.sample_set.values
    )
    samples_match_4w = np.array_equal(
        serial_result.sample_set.values, parallel_4w.sample_set.values
    )
    print(f"  serial vs. 2 workers: identical samples = {samples_match_2w}")
    print(f"  serial vs. 4 workers: identical samples = {samples_match_4w}")
    assert samples_match_2w and samples_match_4w, (
        "Changing worker count must never change the generated uncertainty samples."
    )
    print("Confirmed: the sample set is fully determined before execution, independent "
          "of worker count.")

    serial_outputs = serial_result.output_values(extractor)
    parallel_2w_outputs = parallel_2w.output_values(extractor)
    parallel_4w_outputs = parallel_4w.output_values(extractor)
    outputs_match = np.array_equal(serial_outputs, parallel_2w_outputs) and np.array_equal(
        serial_outputs, parallel_4w_outputs
    )
    print(f"\nVerifying the extracted output values are also bit-identical: {outputs_match}")
    assert outputs_match, (
        "Identical samples through an identical pipeline must produce identical outputs."
    )

    print("\nOutput statistics, computed independently for each run (must all agree):")
    for label, outputs in (
        ("serial", serial_outputs), ("parallel (2w)", parallel_2w_outputs),
        ("parallel (4w)", parallel_4w_outputs),
    ):
        stats = compute_output_statistics(
            outputs, "maximum_displacement", n_requested=config.n_samples,
            n_failed=0, n_invalid=0,
        )
        p50 = stats.percentiles.get(50.0)
        p95 = stats.percentiles.get(95.0)
        print(f"  {label:>16}: n={stats.n_successful}, mean={stats.mean:.6e}, "
              f"std={stats.std:.6e}, P50={p50:.6e}, P95={p95:.6e}")

    print("\nMeasured wall-clock time (this run, this machine -- not a guarantee):")
    print(f"  Serial:          {serial_elapsed:.3f}s")
    print(f"  Parallel (2w):   {parallel_2w_elapsed:.3f}s")
    print(f"  Parallel (4w):   {parallel_4w_elapsed:.3f}s")


if __name__ == "__main__":
    main()
