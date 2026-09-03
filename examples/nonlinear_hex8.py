"""Example: a HEX8 element under Total Lagrangian geometric nonlinearity, Version 16.

Demonstrates the full geometrically nonlinear pipeline for the 8-node
trilinear hexahedron: 2x2x2 Gauss-integrated Total Lagrangian kinematics
(each Gauss point tracking its own deformation gradient and Green-Lagrange
strain independently), a St. Venant-Kirchhoff finite-strain material, and
Newton-Raphson load stepping.
"""

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.materials import LinearElastic3D, SaintVenantKirchhoff3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.results.nonlinear_result import NonlinearAnalysisResult

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
LOAD_PER_NODE = 1.5e7  # N, applied at each of the 4 loaded-face nodes
LOAD_STEPS = 10

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
    """Build, solve, and report a single HEX8 element under large-displacement tension."""
    placeholder_material = LinearElastic3D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, density=7850.0
    )
    finite_strain_material = SaintVenantKirchhoff3D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO
    )

    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder_material)

    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    settings = NonlinearSolverSettings(load_steps=LOAD_STEPS, tolerance=1e-9, max_iterations=40)
    analysis = NonlinearAnalysis(
        mesh, {hexa.id: finite_strain_material}, settings, geometric_nonlinearity=True
    )
    for node_id in _FIXED_FACE:
        for dof in (TranslationDOF.X, TranslationDOF.Y, TranslationDOF.Z):
            analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))
    for node_id in _LOADED_FACE:
        analysis.add_load(NodalLoad(node_id, TranslationDOF.X, LOAD_PER_NODE))

    result = analysis.solve()
    print_summary(mesh, hexa, result)


def print_summary(mesh: Mesh, hexa: Hex8Element3D, result: NonlinearAnalysisResult) -> None:
    """Print a human-readable summary, including each Gauss point's final state."""
    print("Finite Element Toolkit")
    print("Version 16 -- HEX8 Total Lagrangian Geometric Nonlinearity")
    print("=" * 40)

    print(f"\nMesh: {len(mesh.nodes)} nodes, {len(mesh.elements)} HEX8 element(s)")
    print(f"Reference volume: {hexa.volume:.6e} m^3")
    print(f"Converged: {result.converged}")
    print(f"Iterations per load step: {[int(n) for n in result.iteration_counts()]}")

    print("\nFinal nodal displacements:")
    for node in mesh.nodes:
        ux, uy, uz = result.node_displacement(node.id)
        print(f"    Node {node.id}: ux={ux:.6e}, uy={uy:.6e}, uz={uz:.6e} m")

    print("\nFinal per-Gauss-point Green-Lagrange strain (E_xx) and stress (S_xx):")
    for gauss_point in range(8):
        strain = result.element_strain(1, gauss_point=gauss_point)
        stress = result.element_stress(1, gauss_point=gauss_point)
        print(f"    Point {gauss_point}: E_xx={strain[0]:.6e}, S_xx={stress[0]:.6e} Pa")

    total_load = LOAD_PER_NODE * len(_LOADED_FACE)
    nominal_stress = total_load / 1.0  # unit cross-section area
    print(f"\nNominal stress (total load / reference area): {nominal_stress:.6e} Pa")


if __name__ == "__main__":
    main()
