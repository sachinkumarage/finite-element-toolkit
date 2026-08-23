"""Example: Newmark-beta time-history analysis under a sinusoidal load, Version 11.

Demonstrates the full :class:`~femtoolkit.analysis.dynamic_analysis.DynamicAnalysis`
workflow: a fixed-left Q4 plate loaded at its free tip by a
:class:`~femtoolkit.analysis.dynamic_loads.SinusoidalLoad`, integrated
with the Newmark-beta average-acceleration method
(:mod:`femtoolkit.analysis.newmark`).

.. code-block:: text

    Mesh -> DynamicAnalysis (boundary conditions + time-dependent load)
        -> Newmark-beta time-stepping -> displacement/velocity/
           acceleration/reaction history
"""

import numpy as np

from femtoolkit.analysis import BoundaryCondition, TranslationDOF
from femtoolkit.analysis.dynamic_analysis import DynamicAnalysis
from femtoolkit.analysis.dynamic_loads import SinusoidalLoad, TimeDependentNodalLoad
from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, create_quad_mesh

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
DENSITY = 7850.0  # kg/m^3
THICKNESS = 0.01  # m
WIDTH = 0.5  # m
HEIGHT = 0.1  # m
NX = 10
NY = 3
FORCE_AMPLITUDE = 50.0  # N
FORCING_ANGULAR_FREQUENCY = 500.0  # rad/s
TIME_STEP = 2e-5  # s
TOTAL_TIME = 0.02  # s


def main() -> None:
    """Build a cantilevered plate, apply a sinusoidal tip load, and integrate its response."""
    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS,
        poisson_ratio=POISSON_RATIO,
        formulation="plane_stress",
        density=DENSITY,
    )
    domain = Rectangle(width=WIDTH, height=HEIGHT)
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=NX, ny=NY, material=material, thickness=THICKNESS
    )
    tip_node = max((n for n in mesh.nodes if n.x == WIDTH), key=lambda n: n.y)

    analysis = DynamicAnalysis(mesh)
    for node in mesh.nodes_on_boundary(domain.boundary("left")):
        analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.X, 0.0))
        analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.Y, 0.0))

    load = SinusoidalLoad(amplitude=FORCE_AMPLITUDE, angular_frequency=FORCING_ANGULAR_FREQUENCY)
    analysis.add_time_dependent_load(TimeDependentNodalLoad(tip_node.id, TranslationDOF.Y, load))

    result = analysis.solve(time_step=TIME_STEP, total_time=TOTAL_TIME)

    print_summary(mesh, tip_node.id, result)


def print_summary(mesh: Mesh, tip_node_id: int, result) -> None:
    """Print a summary of the tip's displacement time history."""
    print("Finite Element Toolkit")
    print("Version 11 -- Newmark-Beta Dynamic Analysis")
    print("=" * 40)

    print(f"\nMesh:\n    {len(mesh.nodes)} nodes, {len(mesh.elements)} Q4 elements")
    print("\nApplied load (tip, Y direction):")
    print(f"    F(t) = {FORCE_AMPLITUDE} * sin({FORCING_ANGULAR_FREQUENCY} * t) N")
    print(f"\nTime step: {TIME_STEP} s, total time: {TOTAL_TIME} s, steps: {len(result.time) - 1}")

    uy = result.displacement(tip_node_id, TranslationDOF.Y)
    print("\nTip displacement (Y) history (every 200th step):")
    for i in range(0, len(result.time), 200):
        print(f"    t = {result.time[i]:.5f} s, uy = {uy[i]:.6e} m")

    print(f"\nMax |uy|: {abs(uy).max():.6e} m")
    print(f"All results finite: {bool(np.isfinite(result.displacement_history).all())}")


if __name__ == "__main__":
    main()
