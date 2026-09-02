"""Example: a single HEX8 element under uniaxial tension, Version 15.

Demonstrates the Version 15 3D workflow with the 8-node trilinear
hexahedral (HEX8) element: 2x2x2 Gauss-integrated stiffness assembly,
3D boundary conditions, a face load, and the recovered displacement,
representative strain/stress, and von Mises equivalent stress -- again
driven through the existing StaticLinearAnalysis workflow unchanged.
"""

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.results.analysis_result import AnalysisResult

YOUNGS_MODULUS = 200e9  # Pa (steel)
POISSON_RATIO = 0.3
DENSITY = 7850.0  # kg/m^3
FACE_LOAD = 5.0e4  # N, applied to each of the four x=1 face nodes


def main() -> None:
    """Build, solve, and report a single HEX8 unit-cube element under uniaxial tension."""
    material = LinearElastic3D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, density=DENSITY
    )

    coords = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ]
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(coords))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=material)

    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    analysis = StaticLinearAnalysis(mesh)
    fixed_face = (1, 4, 5, 8)  # x = 0 face
    loaded_face = (2, 3, 6, 7)  # x = 1 face
    for node_id in fixed_face:
        for dof in (TranslationDOF.X, TranslationDOF.Y, TranslationDOF.Z):
            analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))
    for node_id in loaded_face:
        analysis.add_load(NodalLoad(node_id, TranslationDOF.X, FACE_LOAD))

    result = analysis.solve()
    print_summary(hexa, result)


def print_summary(hexa: Hex8Element3D, result: AnalysisResult) -> None:
    """Print a human-readable summary of the solved HEX8 element."""
    print("Finite Element Toolkit")
    print("Version 15 -- HEX8 Element Under Uniaxial Tension")
    print("=" * 40)

    print(f"\nElement volume: {hexa.volume:.6e} m^3")

    print("\nNodal displacements:")
    for node in hexa.nodes:
        ux, uy, uz = result.node_displacement(node.id)
        print(f"    Node {node.id}: ux={ux:.6e}, uy={uy:.6e}, uz={uz:.6e} m")

    strain = result.element_strain(1)
    stress = result.element_stress(1)
    von_mises = result.element_von_mises(1)
    hydrostatic = result.element_hydrostatic_stress(1)
    principal = result.element_principal_stresses(1)

    print("\nRepresentative (element-center) strain [xx, yy, zz, xy, yz, xz]:")
    print(f"    {strain}")
    print("\nRepresentative stress [xx, yy, zz, xy, yz, xz] (Pa):")
    print(f"    {stress}")
    print(f"\nVon Mises equivalent stress: {von_mises:.6e} Pa")
    print(f"Hydrostatic (mean) stress:   {hydrostatic:.6e} Pa")
    print(f"Principal stresses (s1 >= s2 >= s3): {principal}")

    expected_sigma_xx = FACE_LOAD * 4 / 1.0  # total force / cross-section area
    print(f"\nExpected sigma_xx (F/A): {expected_sigma_xx:.6e} Pa")


if __name__ == "__main__":
    main()
