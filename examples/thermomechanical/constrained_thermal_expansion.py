"""Example: fully constrained thermal expansion of a HEX8 block, Version 19.

Heats a unit HEX8 block whose every DOF is fixed: no thermal displacement
is possible at all, so the entire thermal eigenstrain becomes mechanical
strain, developing genuine **thermal stress** -- the counterpart to
free_thermal_expansion.py's opposite extreme. Compares directly against
the closed-form fully-triaxially-restrained thermal stress,
``sigma = -E*alpha*dT / (1 - 2*v)``.
"""

from femtoolkit.analysis import BoundaryCondition, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.materials import LinearElastic3D, ThermoelasticMaterial3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.results.nonlinear_result import NonlinearAnalysisResult

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
THERMAL_EXPANSION_COEFFICIENT = 12e-6  # 1/K
REFERENCE_TEMPERATURE = 293.15  # K (20 C)
APPLIED_TEMPERATURE = 393.15  # K (120 C)

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def main() -> None:
    """Build, solve, and report a fully-restrained HEX8 block under uniform heating."""
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

    # Every DOF fixed: the block cannot move at all.
    for node in nodes:
        for dof in (TranslationDOF.X, TranslationDOF.Y, TranslationDOF.Z):
            analysis.add_boundary_condition(BoundaryCondition(node.id, dof, 0.0))

    result = analysis.solve()
    print_summary(mesh, result)


def print_summary(mesh: Mesh, result: NonlinearAnalysisResult) -> None:
    """Print a human-readable summary of the fully-constrained thermal-stress solution."""
    print("Finite Element Toolkit")
    print("Version 19 -- Fully Constrained Thermal Expansion")
    print("=" * 55)

    delta_temperature = APPLIED_TEMPERATURE - REFERENCE_TEMPERATURE
    print(f"\nAlpha = {THERMAL_EXPANSION_COEFFICIENT:.3e} 1/K, dT = {delta_temperature:.1f} K")
    print(f"Converged: {result.converged}")

    print("\nNodal displacements (must all be exactly zero -- fully restrained):")
    for node in mesh.nodes:
        ux, uy, uz = result.node_displacement(node.id)
        print(f"    Node {node.id}: ux={ux:.3e}, uy={uy:.3e}, uz={uz:.3e} m")

    expected_stress = (
        -YOUNGS_MODULUS * THERMAL_EXPANSION_COEFFICIENT * delta_temperature
        / (1.0 - 2.0 * POISSON_RATIO)
    )
    print(f"\nClosed-form thermal stress, -E*alpha*dT/(1-2v): {expected_stress:.6e} Pa")
    print("\nSolved thermal stress at each Gauss point:")
    for gauss_point in range(8):
        state = result.element_state(1, gauss_point=gauss_point)
        print(
            f"    Point {gauss_point}: sigma_xx={state.stress[0]:.6e}, "
            f"sigma_yy={state.stress[1]:.6e}, sigma_zz={state.stress[2]:.6e} Pa"
        )

    print(
        "\nA negative (compressive) stress under heating makes sense: the block "
        "'wants' to expand but cannot, so it is squeezed by its own restrained "
        "thermal eigenstrain, exactly as if it had been mechanically compressed."
    )


if __name__ == "__main__":
    main()
