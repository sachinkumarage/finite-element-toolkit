"""Example: factored load combinations, Version 10.

Demonstrates :class:`~femtoolkit.analysis.load_combination.LoadCombination`:
standard structural engineering practice of combining multiple load
cases with code-prescribed load factors, e.g. an ultimate/strength
combination ``1.2 * Dead + 1.6 * Live``:

.. code-block:: text

    ultimate = LoadCombination(
        name="Ultimate",
        factors={dead_load: 1.2, live_load: 1.6},
    )

Because the underlying structural system is linear, the combination's
result is exactly the same as scaling and summing each load case's own
result (superposition) -- this example verifies that numerically as its
engineering check, using :class:`~femtoolkit.analysis.load_manager.LoadManager`
to solve every load case and the combination together.
"""

from femtoolkit.analysis import DistributedLoad, GravityLoad, LoadCase, LoadCombination, LoadManager
from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, create_quad_mesh
from femtoolkit.results import ResultSet

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
DENSITY = 7850.0  # kg/m^3
THICKNESS = 0.01  # m
WIDTH = 2.0  # m
HEIGHT = 1.0  # m
NX = 4
NY = 2
DEAD_FACTOR = 1.2
LIVE_FACTOR = 1.6


def main() -> None:
    """Build a mesh, combine two load cases with factors, solve, and verify superposition."""
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

    dead_load = LoadCase(name="Dead Load")
    dead_load.fix_boundary(domain.boundary("left"), ux=0.0, uy=0.0)
    dead_load.add_gravity_load(GravityLoad(g=9.81))

    live_load = LoadCase(name="Live Load")
    live_load.fix_boundary(domain.boundary("left"), ux=0.0, uy=0.0)
    live_load.add_distributed_load(DistributedLoad(domain.boundary("top"), magnitude=-20_000.0))

    ultimate = LoadCombination(
        name="Ultimate", factors={dead_load: DEAD_FACTOR, live_load: LIVE_FACTOR}
    )

    manager = LoadManager(mesh)
    manager.add_load_case(dead_load)
    manager.add_load_case(live_load)
    manager.add_combination(ultimate)
    results = manager.solve_all()

    print_summary(mesh, results)


def print_summary(mesh: Mesh, results: ResultSet) -> None:
    """Print reactions for each case/combination and verify superposition holds."""
    print("Finite Element Toolkit")
    print("Version 10 -- Factored Load Combinations")
    print("=" * 40)

    dead_result = results.for_load_case("Dead Load")
    live_result = results.for_load_case("Live Load")
    ultimate_result = results.for_combination("Ultimate")

    def total_reaction(result):
        rx = sum(result.node_reaction(n.id)[0] for n in mesh.nodes)
        ry = sum(result.node_reaction(n.id)[1] for n in mesh.nodes)
        return rx, ry

    dead_rx, dead_ry = total_reaction(dead_result)
    live_rx, live_ry = total_reaction(live_result)
    ultimate_rx, ultimate_ry = total_reaction(ultimate_result)

    print(f"\nDead Load reaction:     Rx = {dead_rx:.3f} N, Ry = {dead_ry:.3f} N")
    print(f"Live Load reaction:     Rx = {live_rx:.3f} N, Ry = {live_ry:.3f} N")
    print(f"\nUltimate = {DEAD_FACTOR}*Dead + {LIVE_FACTOR}*Live:")
    print(f"    Combination reaction:  Rx = {ultimate_rx:.3f} N, Ry = {ultimate_ry:.3f} N")

    expected_rx = DEAD_FACTOR * dead_rx + LIVE_FACTOR * live_rx
    expected_ry = DEAD_FACTOR * dead_ry + LIVE_FACTOR * live_ry
    print(f"    Expected (superposition): Rx = {expected_rx:.3f} N, Ry = {expected_ry:.3f} N")

    matches = abs(ultimate_rx - expected_rx) < 1e-6 and abs(ultimate_ry - expected_ry) < 1e-6
    print(f"\nSuperposition check:\n    {'PASS' if matches else 'FAIL'}")


if __name__ == "__main__":
    main()
