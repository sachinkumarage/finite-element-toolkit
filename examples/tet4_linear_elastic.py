"""Example: a single TET4 element under uniaxial tension, Version 15.

Demonstrates the Version 15 3D workflow: 3D nodes, a 4-node tetrahedral
(TET4) solid element, the 3D isotropic linear elastic material, 3D
boundary conditions, a nodal load, and the recovered displacement, strain,
stress, and von Mises equivalent stress -- driven through the existing
StaticLinearAnalysis workflow unchanged (see the module docstring of
femtoolkit.mesh.tet4_element for why no changes were needed there).
"""

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Mesh, Node, Tet4Element3D
from femtoolkit.results.analysis_result import AnalysisResult

YOUNGS_MODULUS = 70e9  # Pa (aluminum)
POISSON_RATIO = 0.33
DENSITY = 2700.0  # kg/m^3
TIP_LOAD = 1.0e5  # N, applied along X at node 2


def main() -> None:
    """Build, solve, and report a single TET4 element under uniaxial tension."""
    material = LinearElastic3D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, density=DENSITY
    )

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=0.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=0.0, z=1.0)
    tet = Tet4Element3D(id=1, nodes=(node_1, node_2, node_3, node_4), material=material)

    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4):
        mesh.add_node(node)
    mesh.add_element(tet)

    analysis = StaticLinearAnalysis(mesh)
    # Fix nodes 1, 3, 4 completely (a stable, statically determinate support
    # for a single tetrahedron); pull node 2 along X.
    for node_id in (1, 3, 4):
        for dof in (TranslationDOF.X, TranslationDOF.Y, TranslationDOF.Z):
            analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))
    analysis.add_load(NodalLoad(2, TranslationDOF.X, TIP_LOAD))

    result = analysis.solve()
    print_summary(tet, result)


def print_summary(tet: Tet4Element3D, result: AnalysisResult) -> None:
    """Print a human-readable summary of the solved TET4 element."""
    print("Finite Element Toolkit")
    print("Version 15 -- TET4 Element Under Uniaxial Tension")
    print("=" * 40)

    print(f"\nElement volume: {tet.volume:.6e} m^3")

    print("\nNodal displacements:")
    for node in tet.nodes:
        ux, uy, uz = result.node_displacement(node.id)
        print(f"    Node {node.id}: ux={ux:.6e}, uy={uy:.6e}, uz={uz:.6e} m")

    strain = result.element_strain(1)
    stress = result.element_stress(1)
    von_mises = result.element_von_mises(1)
    hydrostatic = result.element_hydrostatic_stress(1)
    principal = result.element_principal_stresses(1)

    print("\nElement strain [xx, yy, zz, xy, yz, xz]:")
    print(f"    {strain}")
    print("\nElement stress [xx, yy, zz, xy, yz, xz] (Pa):")
    print(f"    {stress}")
    print(f"\nVon Mises equivalent stress: {von_mises:.6e} Pa")
    print(f"Hydrostatic (mean) stress:   {hydrostatic:.6e} Pa")
    print(f"Principal stresses (s1 >= s2 >= s3): {principal}")


if __name__ == "__main__":
    main()
