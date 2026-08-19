"""Example: uniform temperature-change loading, Version 10.

Demonstrates :class:`~femtoolkit.analysis.thermal_load.TemperatureLoad`
on the same rectangular plate under two different support conditions,
to show the underlying physics clearly:

* **Free expansion** -- a pin at one corner plus a roller at another
  (the minimal, statically determinate support, same pattern used for
  Version 9's reaction-equilibrium validation): the plate is free to
  expand, so displacement follows ``u = alpha * dT * coordinate``
  exactly and stress is zero everywhere.
* **Fully restrained** -- every node fixed: the plate cannot expand at
  all, so it develops a large, uniform "thermal stress" instead,
  ``sigma = -D @ [alpha*dT, alpha*dT, 0]``.

See :mod:`femtoolkit.analysis.thermal_load` for why the analytically
correct stress requires :func:`~femtoolkit.analysis.thermal_load.thermal_corrected_stress`
rather than the general-purpose ``result.element_stress()``.
"""

from femtoolkit.analysis import BoundaryCondition, StaticLinearAnalysis, TranslationDOF
from femtoolkit.analysis.thermal_load import (
    TemperatureLoad,
    thermal_corrected_stress,
    thermal_load_to_nodal_loads,
)
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, create_quad_mesh

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
ALPHA = 12e-6  # 1/K (structural steel)
THICKNESS = 0.01  # m
WIDTH = 2.0  # m
HEIGHT = 1.0  # m
NX = 4
NY = 2
DELTA_T = 80.0  # K


def build_mesh() -> Mesh:
    """Build a fresh Q4 mesh with a thermally capable material."""
    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS,
        poisson_ratio=POISSON_RATIO,
        formulation="plane_stress",
        thermal_expansion_coefficient=ALPHA,
    )
    return create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=NX, ny=NY, material=material, thickness=THICKNESS
    )


def main() -> None:
    """Solve and compare free-expansion vs. fully-restrained thermal loading."""
    thermal_load = TemperatureLoad(delta_temperature=DELTA_T)

    print("Finite Element Toolkit")
    print("Version 10 -- Uniform Temperature-Change Loading")
    print("=" * 40)
    print(f"\nDelta T = {DELTA_T} K, alpha = {ALPHA} 1/K")

    run_free_expansion(thermal_load)
    run_fully_restrained(thermal_load)


def run_free_expansion(thermal_load: TemperatureLoad) -> None:
    """Pin + roller support: the plate expands freely, stress must be ~0."""
    mesh = build_mesh()
    loads = thermal_load_to_nodal_loads(mesh, thermal_load)

    pin_node = min(mesh.nodes, key=lambda n: (n.x, n.y))
    roller_node = max((n for n in mesh.nodes if n.y == 0.0), key=lambda n: n.x)
    top_right_node = max((n for n in mesh.nodes if n.y == HEIGHT), key=lambda n: n.x)

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(pin_node.id, TranslationDOF.X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(pin_node.id, TranslationDOF.Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(roller_node.id, TranslationDOF.Y, 0.0))
    for load in loads:
        analysis.add_load(load)
    result = analysis.solve()

    print("\nCase 1: Free expansion (pin + roller support)")
    ux, uy = result.node_displacement(top_right_node.id)
    print(f"    Top-right corner displacement: ux = {ux:.6e} m, uy = {uy:.6e} m")
    expected_ux = ALPHA * DELTA_T * WIDTH
    expected_uy = ALPHA * DELTA_T * HEIGHT
    print(f"    Expected: ux = {expected_ux:.6e} m, uy = {expected_uy:.6e} m")
    stress = thermal_corrected_stress(result, mesh.elements[0], thermal_load)
    print(f"    Element 1 thermal-corrected stress: {stress} Pa (expected ~0)")


def run_fully_restrained(thermal_load: TemperatureLoad) -> None:
    """Every node fixed: the plate cannot expand, developing a uniform thermal stress."""
    mesh = build_mesh()
    loads = thermal_load_to_nodal_loads(mesh, thermal_load)

    analysis = StaticLinearAnalysis(mesh)
    for node in mesh.nodes:
        analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.X, 0.0))
        analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.Y, 0.0))
    for load in loads:
        analysis.add_load(load)
    result = analysis.solve()

    material = mesh.elements[0].material
    epsilon_thermal = ALPHA * DELTA_T
    expected = -(material.constitutive_matrix @ [epsilon_thermal, epsilon_thermal, 0.0])

    print("\nCase 2: Fully restrained (every node fixed)")
    stress = thermal_corrected_stress(result, mesh.elements[0], thermal_load)
    print(f"    Element 1 thermal-corrected stress: {stress} Pa")
    print(f"    Expected (-D @ epsilon_thermal):     {expected} Pa")


if __name__ == "__main__":
    main()
