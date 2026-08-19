"""Example: multiple named load cases solved independently, Version 10.

Demonstrates the new Version 10 workflow of building load cases
independently of any mesh -- ``LoadCase(name="Dead Load")``, with no
``mesh=`` argument -- and only binding them to a mesh at solve time via
:class:`~femtoolkit.analysis.load_manager.LoadManager`:

.. code-block:: text

    Geometry -> Mesh -> Load Cases -> Boundary Conditions / Loads
        -> Solver -> Results (one per load case)

Model: a fixed-left plate, once under only its own weight ("Dead Load"),
once under only a distributed load on its top edge ("Live Load"), and
once under only a horizontal traction on its right edge ("Wind Load") --
three completely independent analyses of the same mesh, each queryable
by name from a single :class:`~femtoolkit.results.result_set.ResultSet`.
"""

from femtoolkit.analysis import DistributedLoad, GravityLoad, LoadCase, LoadManager
from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, create_quad_mesh

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
DENSITY = 7850.0  # kg/m^3 (structural steel)
THICKNESS = 0.01  # m
WIDTH = 2.0  # m
HEIGHT = 1.0  # m
NX = 4
NY = 2


def main() -> None:
    """Build a mesh, register three independent load cases, solve them all, and report."""
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

    wind_load = LoadCase(name="Wind Load")
    wind_load.fix_boundary(domain.boundary("left"), ux=0.0, uy=0.0)
    wind_load.add_distributed_load(
        DistributedLoad(domain.boundary("right"), magnitude=5_000.0, direction="global_x")
    )

    manager = LoadManager(mesh)
    manager.add_load_case(dead_load)
    manager.add_load_case(live_load)
    manager.add_load_case(wind_load)
    results = manager.solve_all()

    print_summary(mesh, results)


def print_summary(mesh: Mesh, results) -> None:
    """Print the total left-edge reaction for each independently solved load case."""
    print("Finite Element Toolkit")
    print("Version 10 -- Independent Load Cases")
    print("=" * 40)

    for name in ("Dead Load", "Live Load", "Wind Load"):
        result = results.for_load_case(name)
        total_rx = sum(result.node_reaction(n.id)[0] for n in mesh.nodes)
        total_ry = sum(result.node_reaction(n.id)[1] for n in mesh.nodes)
        print(f"\n{name}:")
        print(f"    Total reaction: Rx = {total_rx:.3f} N, Ry = {total_ry:.3f} N")


if __name__ == "__main__":
    main()
