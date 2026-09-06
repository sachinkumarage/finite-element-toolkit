"""Example: combined mechanical and thermal loading on a HEX8 block, Version 19.

Extends examples/hex8_linear_elastic.py (Version 15) to genuine
thermoelasticity, mirroring thermoelastic_block_tet4.py for the 8-node
solid element: a HEX8 block heated and mechanically loaded together,
reporting each Gauss point's independent stress state.
"""

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.materials import LinearElastic3D, ThermoelasticMaterial3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.results.nonlinear_result import NonlinearAnalysisResult

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
THERMAL_EXPANSION_COEFFICIENT = 12e-6  # 1/K
REFERENCE_TEMPERATURE = 293.15  # K (20 C)
APPLIED_TEMPERATURE = 353.15  # K (80 C)
LOAD_PER_NODE = 2.0e6  # N, applied at each of the 4 loaded-face nodes

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]
_FIXED_FACE = (1, 4, 5, 8)  # x = 0 face
_LOADED_FACE = (2, 3, 6, 7)  # x = 1 face


def main() -> None:
    """Build, solve, and report a HEX8 block under combined mechanical and thermal loading."""
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
    for node_id in _FIXED_FACE:
        for dof in (TranslationDOF.X, TranslationDOF.Y, TranslationDOF.Z):
            analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))
    for node_id in _LOADED_FACE:
        analysis.add_load(NodalLoad(node_id, TranslationDOF.X, LOAD_PER_NODE))

    result = analysis.solve()
    print_summary(mesh, result)


def print_summary(mesh: Mesh, result: NonlinearAnalysisResult) -> None:
    """Print a human-readable summary, including each Gauss point's final stress state."""
    print("Finite Element Toolkit")
    print("Version 19 -- HEX8 Combined Mechanical + Thermal Loading")
    print("=" * 60)

    delta_temperature = APPLIED_TEMPERATURE - REFERENCE_TEMPERATURE
    total_load = LOAD_PER_NODE * len(_LOADED_FACE)
    print(f"\nTotal mechanical load: {total_load:.3e} N, dT = {delta_temperature:.1f} K")
    print(f"Converged: {result.converged}")

    print("\nFinal nodal displacements:")
    for node in mesh.nodes:
        ux, uy, uz = result.node_displacement(node.id)
        print(f"    Node {node.id}: ux={ux:.6e}, uy={uy:.6e}, uz={uz:.6e} m")

    print("\nFinal per-Gauss-point stress (sigma_xx):")
    for gauss_point in range(8):
        state = result.element_state(1, gauss_point=gauss_point)
        print(f"    Point {gauss_point}: sigma_xx={state.stress[0]:.4e} Pa")

    reaction_x_total = sum(result.reaction(node_id, TranslationDOF.X) for node_id in _FIXED_FACE)
    print(
        f"\nTotal reaction (X): {reaction_x_total:.4e} N "
        f"(balances the applied {total_load:.4e} N)"
    )


if __name__ == "__main__":
    main()
