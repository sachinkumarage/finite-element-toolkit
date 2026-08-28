"""Example: modal superposition vs. direct time-history solution, Version 12.

Demonstrates :func:`~femtoolkit.analysis.modal_superposition.modal_superposition`:
a cantilevered Q4 plate under a sinusoidal tip load, solved two ways --
directly (:class:`~femtoolkit.analysis.dynamic_analysis.DynamicAnalysis`,
integrating every free DOF) and via modal superposition (decoupling into
a handful of independent single-DOF modal equations) -- to show that
truncating to a small number of modes still reproduces the direct
solution closely, at a fraction of the number of equations integrated
each step.
"""

import numpy as np

from femtoolkit.analysis import BoundaryCondition, RayleighDamping, TranslationDOF
from femtoolkit.analysis.dynamic_analysis import DynamicAnalysis
from femtoolkit.analysis.dynamic_loads import SinusoidalLoad, TimeDependentNodalLoad
from femtoolkit.analysis.dynamic_system import build_dynamic_system
from femtoolkit.analysis.modal_superposition import modal_superposition
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
FORCE_AMPLITUDE = 60.0  # N
FORCING_ANGULAR_FREQUENCY = 800.0  # rad/s
TIME_STEP = 2e-5  # s
TOTAL_TIME = 0.01  # s
MODES_RETAINED = 6


def main() -> None:
    """Solve the same dynamic system directly and via truncated modal superposition."""
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

    boundary_conditions = []
    for node in mesh.nodes_on_boundary(domain.boundary("left")):
        boundary_conditions.append(BoundaryCondition(node.id, TranslationDOF.X, 0.0))
        boundary_conditions.append(BoundaryCondition(node.id, TranslationDOF.Y, 0.0))

    damping = RayleighDamping(alpha=15.0, beta=0.0001)
    load = TimeDependentNodalLoad(
        tip_node.id,
        TranslationDOF.Y,
        SinusoidalLoad(amplitude=FORCE_AMPLITUDE, angular_frequency=FORCING_ANGULAR_FREQUENCY),
    )

    direct_analysis = DynamicAnalysis(mesh, damping=damping)
    for bc in boundary_conditions:
        direct_analysis.add_boundary_condition(bc)
    direct_analysis.add_time_dependent_load(load)
    direct_result = direct_analysis.solve(time_step=TIME_STEP, total_time=TOTAL_TIME)

    system = build_dynamic_system(mesh, boundary_conditions, damping=damping)
    modal_result = modal_superposition(
        system,
        modes=MODES_RETAINED,
        loads=[load],
        time_step=TIME_STEP,
        total_time=TOTAL_TIME,
    )

    n_free = system.dof_map.total_dofs - len(boundary_conditions)
    print_summary(tip_node.id, direct_result, modal_result, n_free)


def print_summary(tip_node_id: int, direct_result, modal_result, n_free: int) -> None:
    """Print a comparison between the direct and modal-superposition solutions."""
    print("Finite Element Toolkit")
    print("Version 12 -- Modal Superposition vs. Direct Time-History Solution")
    print("=" * 70)

    print(f"\nTotal free DOFs integrated directly: {n_free}")
    print(f"Modes retained for superposition:    {MODES_RETAINED}")

    direct_uy = direct_result.displacement(tip_node_id, TranslationDOF.Y)
    modal_uy = modal_result.displacement(tip_node_id, TranslationDOF.Y)

    max_error = np.abs(modal_uy - direct_uy).max()
    relative_error = max_error / np.abs(direct_uy).max()

    print("\nTip displacement (Y) comparison (every 100th step):")
    print(f"    {'step':>6}{'direct (m)':>16}{'modal (m)':>16}")
    for i in range(0, len(direct_uy), 100):
        print(f"    {i:>6}{direct_uy[i]:>16.6e}{modal_uy[i]:>16.6e}")

    print(f"\nMax absolute error:  {max_error:.6e} m")
    print(f"Max relative error:  {relative_error:.4%}")
    status = "PASS" if relative_error < 0.02 else "FAIL"
    print(f"\n{MODES_RETAINED}-mode approximation quality: {status}")


if __name__ == "__main__":
    main()
