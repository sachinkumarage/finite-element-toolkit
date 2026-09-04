"""Example: a rubber HEX8 block under Total Lagrangian large deformation, Version 17.

Demonstrates femtoolkit.materials.neo_hookean.NeoHookean3D driving the
same Total Lagrangian HEX8 pipeline used in examples/nonlinear_hex8.py
(Version 16), but now genuinely nonlinear: a rubber-like block pulled to
a large uniaxial stretch far beyond the small-strain regime, reporting
the deformation gradient, Jacobian (volume ratio), strain-energy density,
and all three stress measures at each Gauss point via the
HyperelasticMaterial interface, composed with the Version 17 reporting
helper femtoolkit.analysis.geometric_nonlinear.hex8_deformation_gradients.
"""

import numpy as np

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.geometric_nonlinear import hex8_deformation_gradients
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.materials import LinearElastic3D, NeoHookean3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.results.nonlinear_result import NonlinearAnalysisResult

YOUNGS_MODULUS = 5.0e6  # Pa -- soft, rubber-like
POISSON_RATIO = 0.45
LOAD_PER_NODE = 3.0e5  # N, applied at each of the 4 loaded-face nodes
LOAD_STEPS = 25

_COORDS = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.0),
]
_FIXED_FACE = (1, 4, 5, 8)  # x = 0 face
_LOADED_FACE = (2, 3, 6, 7)  # x = 1 face


def main() -> None:
    """Build, solve, and report a rubber HEX8 block under large-displacement tension."""
    placeholder_material = LinearElastic3D(
        youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0
    )
    rubber = NeoHookean3D(youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO)

    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder_material)

    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    settings = NonlinearSolverSettings(load_steps=LOAD_STEPS, tolerance=1e-9, max_iterations=40)
    analysis = NonlinearAnalysis(mesh, {hexa.id: rubber}, settings, geometric_nonlinearity=True)
    for node_id in _FIXED_FACE:
        for dof in (TranslationDOF.X, TranslationDOF.Y, TranslationDOF.Z):
            analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))
    for node_id in _LOADED_FACE:
        analysis.add_load(NodalLoad(node_id, TranslationDOF.X, LOAD_PER_NODE))

    result = analysis.solve()
    print_summary(mesh, hexa, rubber, result)


def print_summary(
    mesh: Mesh, hexa: Hex8Element3D, rubber: NeoHookean3D, result: NonlinearAnalysisResult
) -> None:
    """Print a human-readable summary, including per-Gauss-point kinematics and stress measures."""
    print("Finite Element Toolkit")
    print("Version 17 -- Neo-Hookean Hyperelastic HEX8 Block")
    print("=" * 60)

    print(f"\nE = {YOUNGS_MODULUS:.3e} Pa, v = {POISSON_RATIO}")
    print(f"mu (shear modulus) = {rubber.shear_modulus:.4e} Pa")
    print(f"lambda (first Lame parameter) = {rubber.lame_lambda:.4e} Pa")
    print(f"\nMesh: {len(mesh.nodes)} nodes, {len(mesh.elements)} HEX8 element(s)")
    print(f"Converged: {result.converged}")
    print(f"Iterations per load step: {[int(n) for n in result.iteration_counts()]}")

    print("\nFinal nodal displacements:")
    for node in mesh.nodes:
        ux, uy, uz = result.node_displacement(node.id)
        print(f"    Node {node.id}: ux={ux:.6e}, uy={uy:.6e}, uz={uz:.6e} m")

    final_displacements = []
    for node in mesh.nodes:
        final_displacements.extend(result.node_displacement(node.id))
    deformation_gradients = hex8_deformation_gradients(hexa, final_displacements)

    print("\nPer-Gauss-point kinematics and stress at the final load step:")
    for gauss_point, deformation_gradient_tensor in enumerate(deformation_gradients):
        jacobian = float(np.linalg.det(deformation_gradient_tensor))
        energy_density = rubber.strain_energy_density(deformation_gradient_tensor)
        cauchy_stress = rubber.cauchy_stress(deformation_gradient_tensor)
        print(
            f"    Point {gauss_point}: F_xx={deformation_gradient_tensor[0, 0]:.4f}, "
            f"J~{jacobian:.4f}, W={energy_density:.4e} J/m^3, "
            f"sigma_xx={cauchy_stress[0, 0]:.4e} Pa"
        )

    total_load = LOAD_PER_NODE * len(_LOADED_FACE)
    nominal_stress = total_load / 1.0  # unit reference cross-section area
    print(f"\nNominal reference-area stress (total load / reference area): {nominal_stress:.4e} Pa")


if __name__ == "__main__":
    main()
