"""Example: free thermal expansion of a HEX8 block, Version 19.

Heats a unit HEX8 block that is only minimally constrained (just enough
to remove rigid-body modes, not to resist expansion in any direction),
demonstrating the defining property of *free* thermal expansion: genuine
thermal deformation develops (matching the closed-form
``u = alpha*dT*X`` exactly), while the mechanical stress stays
essentially zero -- the body is free to accommodate the thermal
eigenstrain without developing any internal resistance.
"""

from femtoolkit.analysis import BoundaryCondition, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.materials import LinearElastic3D, ThermoelasticMaterial3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.results.nonlinear_result import NonlinearAnalysisResult

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
THERMAL_EXPANSION_COEFFICIENT = 12e-6  # 1/K, typical structural steel
REFERENCE_TEMPERATURE = 293.15  # K (20 C)
APPLIED_TEMPERATURE = 393.15  # K (120 C)

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def main() -> None:
    """Build, solve, and report a freely-expanding HEX8 block under uniform heating."""
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

    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder_material)

    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    material = thermoelastic_material.at_temperature(APPLIED_TEMPERATURE)
    settings = NonlinearSolverSettings(load_steps=1, tolerance=1e-6, max_iterations=15)
    analysis = NonlinearAnalysis(mesh, {hexa.id: material}, settings, geometric_nonlinearity=False)

    # Minimal, statically-determinate constraints: remove rigid-body modes only,
    # leaving the block entirely free to expand.
    analysis.add_boundary_condition(BoundaryCondition(1, TranslationDOF.X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, TranslationDOF.Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, TranslationDOF.Z, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(2, TranslationDOF.Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(2, TranslationDOF.Z, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, TranslationDOF.Z, 0.0))

    result = analysis.solve()
    print_summary(mesh, result)


def print_summary(mesh: Mesh, result: NonlinearAnalysisResult) -> None:
    """Print a human-readable summary of the free-expansion solution."""
    print("Finite Element Toolkit")
    print("Version 19 -- Free Thermal Expansion")
    print("=" * 45)

    delta_temperature = APPLIED_TEMPERATURE - REFERENCE_TEMPERATURE
    print(f"\nAlpha = {THERMAL_EXPANSION_COEFFICIENT:.3e} 1/K, dT = {delta_temperature:.1f} K")
    print(f"Converged: {result.converged}")

    print("\nNodal displacements (compare to u = alpha*dT*X):")
    for node in mesh.nodes:
        ux, uy, uz = result.node_displacement(node.id)
        expected_ux = THERMAL_EXPANSION_COEFFICIENT * delta_temperature * node.x
        print(
            f"    Node {node.id} at X={node.x:.1f}: ux={ux:.6e} m "
            f"(expected {expected_ux:.6e} m), uy={uy:.6e}, uz={uz:.6e}"
        )

    print("\nMechanical stress at each Gauss point (should be ~0 Pa):")
    for gauss_point in range(8):
        state = result.element_state(1, gauss_point=gauss_point)
        print(f"    Point {gauss_point}: max|sigma|={max(abs(state.stress)):.4e} Pa")


if __name__ == "__main__":
    main()
