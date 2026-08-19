"""Example: self-weight (gravity) loading of a cantilevered plate, Version 10.

Demonstrates :class:`~femtoolkit.analysis.body_load.GravityLoad`: a
fixed-left plate loaded by nothing but its own weight. Unlike a
distributed boundary traction (Version 9), gravity is a **body force**
acting throughout every element's area, not along one edge -- see
:mod:`femtoolkit.analysis.body_load` for the equivalent-nodal-force
math (an exact 1/3 split per CST element, 2x2 Gauss quadrature per Q4
element).

The total vertical reaction must equal the plate's total weight exactly
(``W = density * width * height * thickness * g``), independent of mesh
density -- the engineering validation check this example reports.
"""

from femtoolkit.analysis import GravityLoad, LoadCase
from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, create_quad_mesh
from femtoolkit.results import AnalysisResult

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
DENSITY = 7850.0  # kg/m^3 (structural steel)
THICKNESS = 0.01  # m
WIDTH = 2.0  # m
HEIGHT = 1.0  # m
NX = 4
NY = 2
G = 9.81  # m/s^2
EQUILIBRIUM_TOLERANCE = 1e-6  # N


def main() -> None:
    """Build, solve, and report a plate under only its own weight."""
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

    load_case = LoadCase(name="Self Weight", mesh=mesh)
    load_case.fix_boundary(domain.boundary("left"), ux=0.0, uy=0.0)
    load_case.add_gravity_load(GravityLoad(g=G))
    result = load_case.solve()

    print_summary(mesh, result)


def print_summary(mesh: Mesh, result: AnalysisResult) -> None:
    """Print reaction totals and the total-weight equilibrium check."""
    print("Finite Element Toolkit")
    print("Version 10 -- Self-Weight (Gravity) Loading")
    print("=" * 40)

    print(f"\nDomain:\n    {WIDTH} m x {HEIGHT} m x {THICKNESS} m, density = {DENSITY} kg/m^3")

    total_rx = sum(result.node_reaction(n.id)[0] for n in mesh.nodes)
    total_ry = sum(result.node_reaction(n.id)[1] for n in mesh.nodes)
    print(f"\nTotal reaction:\n    Rx = {total_rx:.6f} N, Ry = {total_ry:.6f} N")

    expected_weight = DENSITY * (WIDTH * HEIGHT) * THICKNESS * G
    print(f"\nExpected total weight (density * area * thickness * g):\n    {expected_weight:.6f} N")

    equilibrium_ok = (
        abs(total_ry - expected_weight) < EQUILIBRIUM_TOLERANCE
        and abs(total_rx) < EQUILIBRIUM_TOLERANCE
    )
    print(f"\nGlobal Equilibrium:\n    {'PASS' if equilibrium_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
