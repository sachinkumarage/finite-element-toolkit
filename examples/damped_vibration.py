"""Example: Rayleigh damping and decaying vibration, Version 11.

Demonstrates :class:`~femtoolkit.analysis.damping.RayleighDamping`
(``C = alpha*M + beta*K``) by comparing an undamped and a damped
cantilevered plate's response to the same suddenly applied tip load:
the undamped case oscillates indefinitely at constant amplitude, while
the damped case's oscillation amplitude decays over time.
"""

import numpy as np

from femtoolkit.analysis import BoundaryCondition, TranslationDOF
from femtoolkit.analysis.damping import RayleighDamping
from femtoolkit.analysis.dynamic_analysis import DynamicAnalysis
from femtoolkit.analysis.dynamic_loads import ConstantLoad, TimeDependentNodalLoad
from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import create_quad_mesh

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
DENSITY = 7850.0  # kg/m^3
THICKNESS = 0.01  # m
WIDTH = 0.5  # m
HEIGHT = 0.1  # m
NX = 10
NY = 3
TIP_FORCE = -50.0  # N
TIME_STEP = 2e-5  # s
TOTAL_TIME = 0.05  # s
RAYLEIGH_ALPHA = 200.0  # mass-proportional damping coefficient, 1/s


def run(damping: RayleighDamping | None):
    """Build the same cantilevered plate and solve it with the given damping."""
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

    analysis = DynamicAnalysis(mesh, damping=damping)
    for node in mesh.nodes_on_boundary(domain.boundary("left")):
        analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.X, 0.0))
        analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.Y, 0.0))
    analysis.add_time_dependent_load(
        TimeDependentNodalLoad(tip_node.id, TranslationDOF.Y, ConstantLoad(TIP_FORCE))
    )

    result = analysis.solve(time_step=TIME_STEP, total_time=TOTAL_TIME)
    return result.displacement(tip_node.id, TranslationDOF.Y)


def main() -> None:
    """Compare undamped and Rayleigh-damped response to the same step load."""
    undamped_uy = run(damping=None)
    damped_uy = run(damping=RayleighDamping(alpha=RAYLEIGH_ALPHA, beta=0.0))

    print("Finite Element Toolkit")
    print("Version 11 -- Rayleigh Damping and Decaying Vibration")
    print("=" * 40)

    print(f"\nDamping: alpha = {RAYLEIGH_ALPHA} 1/s, beta = 0.0 s  (C = alpha * M)")

    n_steps = len(undamped_uy)
    quarter = n_steps // 4

    undamped_first = np.abs(undamped_uy[:quarter]).max()
    undamped_last = np.abs(undamped_uy[-quarter:]).max()
    damped_first = np.abs(damped_uy[:quarter]).max()
    damped_last = np.abs(damped_uy[-quarter:]).max()

    print("\nPeak |tip displacement|, first quarter vs. last quarter of the simulation:")
    print(f"    Undamped: {undamped_first:.6e} m -> {undamped_last:.6e} m")
    print(f"    Damped:   {damped_first:.6e} m -> {damped_last:.6e} m")

    undamped_ratio = undamped_last / undamped_first
    damped_ratio = damped_last / damped_first
    print("\nAmplitude ratio (last/first quarter):")
    print(f"    Undamped: {undamped_ratio:.4f}  (expected: close to 1.0, no decay)")
    print(f"    Damped:   {damped_ratio:.4f}  (expected: well below 1.0, decaying)")

    decaying = damped_ratio < undamped_ratio
    print(f"\nDamping causes visible decay: {'PASS' if decaying else 'FAIL'}")


if __name__ == "__main__":
    main()
