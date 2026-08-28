"""Example: FEM modal analysis with participation factors and effective modal mass, Version 12.

Extends Version 11's natural-frequency analysis
(``examples/natural_frequency_analysis.py``) with the Version 12 modal
mass summary: for each mode, its participation factor, effective modal
mass, and cumulative effective mass ratio in the global X and Y
directions -- the standard "how much of the structure's mass does each
mode actually move" summary used to decide how many modes matter for a
given direction of shaking.
"""

from femtoolkit.analysis import BoundaryCondition, TranslationDOF
from femtoolkit.analysis.dynamic_system import build_dynamic_system
from femtoolkit.analysis.modal import modal_analysis_of_system
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
NUM_MODES = 6


def main() -> None:
    """Build a cantilevered Q4 plate and report its modal mass summary in X and Y."""
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

    boundary_conditions = []
    for node in mesh.nodes_on_boundary(domain.boundary("left")):
        boundary_conditions.append(BoundaryCondition(node.id, TranslationDOF.X, 0.0))
        boundary_conditions.append(BoundaryCondition(node.id, TranslationDOF.Y, 0.0))

    system = build_dynamic_system(mesh, boundary_conditions)

    result_x = modal_analysis_of_system(system, num_modes=NUM_MODES, direction="x")
    result_y = modal_analysis_of_system(system, num_modes=NUM_MODES, direction="y")

    print_summary(mesh, result_x, result_y)


def print_summary(mesh: Mesh, result_x, result_y) -> None:
    """Print natural frequencies, periods, and the modal mass summary table."""
    print("Finite Element Toolkit")
    print("Version 12 -- Modal Analysis with Participation Factors")
    print("=" * 70)

    print(f"\nMesh:\n    {len(mesh.nodes)} nodes, {len(mesh.elements)} Q4 elements")

    print("\nNatural frequencies and periods:")
    for i in range(NUM_MODES):
        freq, period = result_x.frequencies[i], result_x.periods[i]
        print(f"    Mode {i + 1}: f = {freq:8.3f} Hz, T = {period:.6f} s")

    header = (
        f"    {'Mode':<6}{'Freq (Hz)':>12}{'Participation':>16}"
        f"{'Eff. Mass (kg)':>18}{'Cum. Ratio':>14}"
    )

    print("\nModal mass summary -- X direction:")
    print(header)
    for i in range(NUM_MODES):
        print(
            f"    {i + 1:<6}{result_x.frequencies[i]:>12.3f}"
            f"{result_x.participation_factors[i]:>16.4f}"
            f"{result_x.effective_modal_mass[i]:>18.6f}"
            f"{result_x.cumulative_mass_ratio[i]:>14.4f}"
        )

    print("\nModal mass summary -- Y direction:")
    print(header)
    for i in range(NUM_MODES):
        print(
            f"    {i + 1:<6}{result_y.frequencies[i]:>12.3f}"
            f"{result_y.participation_factors[i]:>16.4f}"
            f"{result_y.effective_modal_mass[i]:>18.6f}"
            f"{result_y.cumulative_mass_ratio[i]:>14.4f}"
        )

    cumulative_x = result_x.cumulative_mass_ratio[-1]
    cumulative_y = result_y.cumulative_mass_ratio[-1]
    print(f"\nCumulative mass ratio after {NUM_MODES} modes (X): {cumulative_x:.4f}")
    print(f"Cumulative mass ratio after {NUM_MODES} modes (Y): {cumulative_y:.4f}")
    print("(approaches 1.0 as more modes are included -- see the engineering validation suite)")


if __name__ == "__main__":
    main()
