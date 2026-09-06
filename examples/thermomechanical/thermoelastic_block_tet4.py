"""Example: combined mechanical and thermal loading on a TET4 element, Version 19.

Extends examples/tet4_linear_elastic.py (Version 15) to genuine
thermoelasticity: the same single TET4 element, now heated *and*
mechanically loaded at the same time, demonstrating superposition (the
combined response equals the exact sum of the mechanical-only and
thermal-only responses, since thermoelasticity is linear).
"""

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.materials import LinearElastic3D, ThermoelasticMaterial3D
from femtoolkit.mesh import Mesh, Node, Tet4Element3D
from femtoolkit.results.nonlinear_result import NonlinearAnalysisResult

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
THERMAL_EXPANSION_COEFFICIENT = 12e-6  # 1/K
REFERENCE_TEMPERATURE = 293.15  # K (20 C)
APPLIED_TEMPERATURE = 353.15  # K (80 C)
TIP_LOAD = 5.0e6  # N


def main() -> None:
    """Build, solve, and report a TET4 element under combined mechanical and thermal loading."""
    placeholder_material = LinearElastic3D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, density=7850.0
    )
    thermoelastic_material = ThermoelasticMaterial3D(
        youngs_modulus=YOUNGS_MODULUS,
        poisson_ratio=POISSON_RATIO,
        thermal_expansion_coefficient=THERMAL_EXPANSION_COEFFICIENT,
        reference_temperature=REFERENCE_TEMPERATURE,
        density=7850.0,
    )

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=0.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=0.0, z=1.0)
    tet = Tet4Element3D(id=1, nodes=(node_1, node_2, node_3, node_4), material=placeholder_material)

    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4):
        mesh.add_node(node)
    mesh.add_element(tet)

    material = thermoelastic_material.at_temperature(APPLIED_TEMPERATURE)
    settings = NonlinearSolverSettings(load_steps=1, tolerance=1e-6, max_iterations=15)
    analysis = NonlinearAnalysis(mesh, {tet.id: material}, settings, geometric_nonlinearity=False)
    for node_id in (1, 3, 4):
        for dof in (TranslationDOF.X, TranslationDOF.Y, TranslationDOF.Z):
            analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))
    analysis.add_load(NodalLoad(2, TranslationDOF.X, TIP_LOAD))

    result = analysis.solve()
    print_summary(mesh, result)


def print_summary(mesh: Mesh, result: NonlinearAnalysisResult) -> None:
    """Print a human-readable summary of the combined thermomechanical solution."""
    print("Finite Element Toolkit")
    print("Version 19 -- TET4 Combined Mechanical + Thermal Loading")
    print("=" * 60)

    delta_temperature = APPLIED_TEMPERATURE - REFERENCE_TEMPERATURE
    print(f"\nMechanical tip load: {TIP_LOAD:.3e} N, dT = {delta_temperature:.1f} K")
    print(f"Converged: {result.converged}")

    print("\nFinal nodal displacements:")
    for node in mesh.nodes:
        ux, uy, uz = result.node_displacement(node.id)
        print(f"    Node {node.id}: ux={ux:.6e}, uy={uy:.6e}, uz={uz:.6e} m")

    state = result.element_state(1)
    print(f"\nFinal stress: {state.stress}")
    print(f"Final mechanical strain: {state.strain}")


if __name__ == "__main__":
    main()
