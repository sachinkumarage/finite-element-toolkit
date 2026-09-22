"""Example: element-computation scaling study, Version 27.

Measures serial vs. parallel element-stiffness computation time across
several mesh sizes (small, medium, large), and reports speedup and
parallel efficiency (spec section 21) at each size. This is where a
genuine, if modest, parallel speedup can appear for this toolkit's Q4
element -- unlike the single-size comparison in ``serial_vs_parallel.py``,
which uses a mesh small enough that per-task overhead dominates.

Hardware affects every number in this script: core count, per-core
clock speed, and OS process-scheduling overhead all vary between
machines, so re-running this example on a different computer will print
different numbers -- do not treat the printed values as guaranteed. What
should hold qualitatively, per Amdahl's Law
(:func:`femtoolkit.performance.amdahl_speedup`), is that speedup
improves (or at least does not get worse) as the workload grows relative
to the fixed per-task dispatch overhead.
"""

from __future__ import annotations

from femtoolkit.analysis.parallel_assembly import compute_stiffness_contributions
from femtoolkit.execution import ExecutionConfig, create_executor
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh.generator import create_quad_mesh
from femtoolkit.performance import ELEMENT, Profiler, parallel_efficiency, speedup

_WORKERS = 4

_MESH_SIZES = [
    ("Small", 20, 5),
    ("Medium", 60, 15),
    ("Large", 300, 60),
]


def _element_time(elements, config: ExecutionConfig) -> float:
    executor = create_executor(config)
    profiler = Profiler()
    with profiler, profiler.stage(ELEMENT):
        compute_stiffness_contributions(elements, executor)
    return profiler.report().element_time


def main() -> None:
    """Measure serial vs. parallel element-computation time across three mesh sizes."""
    print("Finite Element Toolkit")
    print("Version 27 -- Element-Computation Scaling Study")
    print("=" * 50)

    material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    parallel_config = ExecutionConfig(mode="parallel", workers=_WORKERS, backend="process")

    print(f"\n{'Model':<8}{'Elements':>10}{'Serial (s)':>14}{'Parallel (s)':>16}")
    print(f"{'':<8}{'':>10}{'':>14}{'':>16}{'Speedup':>10}{'Efficiency':>12}")
    print("-" * 70)

    for label, nx, ny in _MESH_SIZES:
        mesh = create_quad_mesh(
            width=float(nx), height=float(ny), nx=nx, ny=ny, material=material, thickness=0.02
        )
        serial_time = _element_time(mesh.elements, ExecutionConfig(mode="serial"))
        parallel_time = _element_time(mesh.elements, parallel_config)
        run_speedup = speedup(serial_time, parallel_time)
        run_efficiency = parallel_efficiency(run_speedup, _WORKERS)

        print(
            f"{label:<8}{len(mesh.elements):>10}{serial_time:>14.4f}{parallel_time:>16.4f}"
            f"{run_speedup:>10.2f}{run_efficiency:>12.2f}"
        )

    print(
        "\nNote: speedup and efficiency above are this run's actual, unmodified "
        "measurements on this machine -- re-running on different hardware, or under "
        "different system load, will produce different numbers. Efficiency well below "
        "1.0 reflects real overhead (worker dispatch, data serialization for the process "
        "backend), not a bug -- it is exactly what Amdahl's Law predicts when the "
        "parallelizable fraction of the work is small relative to fixed per-task cost."
    )


if __name__ == "__main__":
    main()
