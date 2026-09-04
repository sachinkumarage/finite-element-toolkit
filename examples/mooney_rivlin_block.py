"""Example: a 2-parameter Mooney-Rivlin rubber HEX8 block, Version 17.

Demonstrates femtoolkit.materials.mooney_rivlin.MooneyRivlin3D -- a
richer isotropic hyperelastic model than Neo-Hookean, adding a second
invariant-dependent term -- driving the same Total Lagrangian HEX8
pipeline as examples/neo_hookean_block.py, and additionally reports the
Jacobian ``J`` at every Gauss point as a direct, practical check that the
volumetric penalty term is keeping the block close to incompressible
under a stiff ``bulk_modulus`` (see
tests/validation/test_nearly_incompressible.py for the formal version of
this check).
"""

import numpy as np

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.geometric_nonlinear import hex8_deformation_gradients
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.materials import LinearElastic3D, MooneyRivlin3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.results.nonlinear_result import NonlinearAnalysisResult

C10 = 0.4e6  # Pa
C01 = 0.1e6  # Pa
BULK_MODULUS = 2000e6  # Pa -- stiff relative to C10+C01: nearly incompressible
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
    """Build, solve, and report a nearly incompressible Mooney-Rivlin HEX8 block."""
    placeholder_material = LinearElastic3D(
        youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0
    )
    rubber = MooneyRivlin3D(c10=C10, c01=C01, bulk_modulus=BULK_MODULUS)

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
    mesh: Mesh, hexa: Hex8Element3D, rubber: MooneyRivlin3D, result: NonlinearAnalysisResult
) -> None:
    """Print a human-readable summary, emphasizing the near-incompressibility check."""
    print("Finite Element Toolkit")
    print("Version 17 -- Mooney-Rivlin Hyperelastic HEX8 Block")
    print("=" * 60)

    print(f"\nC10 = {C10:.3e} Pa, C01 = {C01:.3e} Pa, K = {BULK_MODULUS:.3e} Pa")
    print(f"Initial shear modulus (2*(C10+C01)) = {rubber.initial_shear_modulus:.4e} Pa")
    print(f"K / (2*(C10+C01)) = {BULK_MODULUS / rubber.initial_shear_modulus:.1f} "
          "(near-incompressibility ratio)")
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

    print("\nPer-Gauss-point kinematics (checking J stays close to 1):")
    for gauss_point, deformation_gradient_tensor in enumerate(deformation_gradients):
        jacobian = float(np.linalg.det(deformation_gradient_tensor))
        energy_density = rubber.strain_energy_density(deformation_gradient_tensor)
        cauchy_stress = rubber.cauchy_stress(deformation_gradient_tensor)
        print(
            f"    Point {gauss_point}: F_xx={deformation_gradient_tensor[0, 0]:.4f}, "
            f"J={jacobian:.6f}, W={energy_density:.4e} J/m^3, "
            f"sigma_xx={cauchy_stress[0, 0]:.4e} Pa"
        )

    max_volume_deviation = max(
        abs(float(np.linalg.det(f)) - 1.0) for f in deformation_gradients
    )
    print(f"\nMax |J-1| across all Gauss points: {max_volume_deviation:.4e} (near-incompressible)")


if __name__ == "__main__":
    main()
