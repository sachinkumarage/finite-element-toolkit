"""Example: a TET4 element under Total Lagrangian geometric nonlinearity, Version 16.

Demonstrates the full geometrically nonlinear pipeline for a 3D solid:
Total Lagrangian TET4 kinematics (deformation gradient, Green-Lagrange
strain), a St. Venant-Kirchhoff finite-strain material, and Newton-Raphson
load stepping through femtoolkit.analysis.nonlinear_analysis.NonlinearAnalysis
with geometric_nonlinearity=True.
"""

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.continuum.stress import cauchy_stress_from_second_piola_kirchhoff
from femtoolkit.continuum.tensor import voigt_stress_to_tensor
from femtoolkit.materials import LinearElastic3D, SaintVenantKirchhoff3D
from femtoolkit.mesh import Mesh, Node, Tet4Element3D
from femtoolkit.results.nonlinear_result import NonlinearAnalysisResult

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
TIP_LOAD = 3.0e7  # N, applied along X at node 2
LOAD_STEPS = 10


def main() -> None:
    """Build, solve, and report a single TET4 element under large-displacement tension."""
    # The element's own `material` is a placeholder LinearElastic3D (elements
    # always carry a linear material); the finite-strain material is supplied
    # separately to NonlinearAnalysis, exactly like Version 15's J2 plasticity
    # examples do for the small-strain nonlinear pathway.
    placeholder_material = LinearElastic3D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, density=7850.0
    )
    finite_strain_material = SaintVenantKirchhoff3D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO
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

    settings = NonlinearSolverSettings(load_steps=LOAD_STEPS, tolerance=1e-9, max_iterations=40)
    analysis = NonlinearAnalysis(
        mesh, {tet.id: finite_strain_material}, settings, geometric_nonlinearity=True
    )
    for node_id in (1, 3, 4):
        for dof in (TranslationDOF.X, TranslationDOF.Y, TranslationDOF.Z):
            analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))
    analysis.add_load(NodalLoad(2, TranslationDOF.X, TIP_LOAD))

    result = analysis.solve()
    print_summary(mesh, tet, result)


def print_summary(mesh: Mesh, tet: Tet4Element3D, result: NonlinearAnalysisResult) -> None:
    """Print a human-readable summary of the load-stepped Total Lagrangian solution."""
    print("Finite Element Toolkit")
    print("Version 16 -- TET4 Total Lagrangian Geometric Nonlinearity")
    print("=" * 40)

    print(f"\nMesh: {len(mesh.nodes)} nodes, {len(mesh.elements)} TET4 element(s)")
    print(f"Reference volume: {tet.volume:.6e} m^3")
    print(f"Converged: {result.converged}")
    print(f"Iterations per load step: {[int(n) for n in result.iteration_counts()]}")

    print("\nFinal nodal displacements:")
    for node in mesh.nodes:
        ux, uy, uz = result.node_displacement(node.id)
        print(f"    Node {node.id}: ux={ux:.6e}, uy={uy:.6e}, uz={uz:.6e} m")

    final_strain = result.element_strain(1)
    final_stress = result.element_stress(1)
    print(f"\nFinal Green-Lagrange strain [xx,yy,zz,2xy,2yz,2xz]:\n    {final_strain}")
    print(f"Final second Piola-Kirchhoff stress [xx,yy,zz,xy,yz,xz] (Pa):\n    {final_stress}")

    # Recover the physically intuitive Cauchy (true) stress for reporting, via
    # the deformation gradient at the converged displacement -- recomputed
    # here from the same public continuum utilities the solver itself uses
    # internally (reference-configuration shape gradients + current
    # displacement), not stored directly in the result (see the README's
    # Version 16 section for why).
    import numpy as np

    from femtoolkit.continuum.deformation import deformation_gradient, displacement_gradient
    from femtoolkit.continuum.jacobian import jacobian_matrix_3d
    from femtoolkit.continuum.shape_functions import tet4_shape_function_derivatives

    dn_dxi, dn_deta, dn_dzeta = tet4_shape_function_derivatives()
    x_coords = tuple(node.x for node in mesh.nodes)
    y_coords = tuple(node.y for node in mesh.nodes)
    z_coords = tuple(node.z for node in mesh.nodes)
    jacobian = jacobian_matrix_3d(dn_dxi, dn_deta, dn_dzeta, x_coords, y_coords, z_coords)
    natural_derivatives = np.array([dn_dxi, dn_deta, dn_dzeta])
    reference_gradients = (np.linalg.inv(jacobian) @ natural_derivatives).T

    displacements = [
        result.displacement(node.id, dof)
        for node in mesh.nodes
        for dof in (TranslationDOF.X, TranslationDOF.Y, TranslationDOF.Z)
    ]
    deformation_gradient_tensor = deformation_gradient(
        displacement_gradient(displacements, reference_gradients)
    )
    cauchy_stress = cauchy_stress_from_second_piola_kirchhoff(
        deformation_gradient_tensor, voigt_stress_to_tensor(final_stress)
    )
    print(f"\nCauchy (true) stress tensor (Pa):\n{cauchy_stress}")


if __name__ == "__main__":
    main()
