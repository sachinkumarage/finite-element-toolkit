"""Example: global force equilibrium and thermal energy balance checks (Version 29).

**Engineering problem/procedure.** A cantilever plate (mechanical) and a
1D conduction bar (thermal) are each solved, then checked against a
first-principles conservation law:
:func:`~femtoolkit.verification.checks.check_force_equilibrium` (applied
loads + reactions = 0, per direction) and
:func:`~femtoolkit.verification.checks.check_thermal_energy_balance`
(supplied heat flow = removed heat flow). A correctly modeled,
correctly solved system passes both by construction; the mechanical
example additionally shows what an artificially corrupted reaction
vector reports, to demonstrate the check actually detects a real
imbalance.
"""

from __future__ import annotations

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.analysis.system import build_force_vector
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh.generator import create_quad_mesh
from femtoolkit.thermal import PrescribedTemperature, SteadyStateThermalAnalysis, ThermalMaterial
from femtoolkit.verification.checks import check_force_equilibrium, check_thermal_energy_balance

X = TranslationDOF.X
Y = TranslationDOF.Y


def main() -> None:
    """Run force equilibrium and thermal energy balance checks on real solved models."""
    print("Finite Element Toolkit")
    print("Version 29 -- Equilibrium Checks")
    print("=" * 35)

    material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    mesh = create_quad_mesh(width=4.0, height=1.0, nx=15, ny=4, material=material, thickness=0.02)
    analysis = StaticLinearAnalysis(mesh)
    for node in mesh.nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(BoundaryCondition(node.id, X, 0.0))
            analysis.add_boundary_condition(BoundaryCondition(node.id, Y, 0.0))
    loads = [NodalLoad(n.id, Y, -10000.0) for n in mesh.nodes if n.x == 4.0 and n.y == 0.0]
    for load in loads:
        analysis.add_load(load)
    result = analysis.solve()
    forces = build_force_vector(result.dof_map, loads)

    check = check_force_equilibrium(result.dof_map, forces, result.reactions)
    print(f"\n{check.name}: {check.message}")

    corrupted_reactions = result.reactions.copy()
    corrupted_reactions[0] += 1e6
    corrupted_check = check_force_equilibrium(result.dof_map, forces, corrupted_reactions)
    print(f"\nArtificially corrupted reactions: {corrupted_check.message}")

    dummy = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    thermal_mesh = create_quad_mesh(
        width=1.0, height=1.0, nx=8, ny=8, material=dummy, thickness=0.01
    )
    thermal_material = ThermalMaterial(
        thermal_conductivity=45.0, density=7850.0, specific_heat=460.0
    )
    materials = {e.id: thermal_material for e in thermal_mesh.elements}
    thermal_analysis = SteadyStateThermalAnalysis(thermal_mesh, materials)
    for node in thermal_mesh.nodes:
        if node.x == 0.0:
            thermal_analysis.add_boundary_condition(PrescribedTemperature(node.id, 373.15))
        elif node.x == 1.0:
            thermal_analysis.add_boundary_condition(PrescribedTemperature(node.id, 293.15))
    thermal_result = thermal_analysis.solve()

    thermal_check = check_thermal_energy_balance(thermal_analysis, thermal_result)
    print(f"\n{thermal_check.name}: {thermal_check.message}")


if __name__ == "__main__":
    main()
