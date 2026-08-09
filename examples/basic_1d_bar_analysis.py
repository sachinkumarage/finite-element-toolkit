"""Example: solve a minimal 1D axial bar problem with Version 3.

This script demonstrates the complete Version 3 structural analysis
workflow:

    Material -> Cross-section -> Nodes -> Bar element -> Mesh
        -> Boundary conditions -> Loads -> Analysis -> Solution -> Results
        -> Analytical validation

Model:

    Node 1 -------- Node 2
           Bar Element

    E = 200 GPa, A = 0.01 m^2, L = 2 m
    Node 1: fixed (u = 0)
    Node 2: F = 1000 N

The computed displacement, reaction, strain, stress, and axial force are
reported, and the displacement is checked against the analytical solution
for a fixed-free axial bar, u = F * L / (E * A).
"""

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis
from femtoolkit.materials import Material
from femtoolkit.mesh import BarElement, Mesh, Node
from femtoolkit.sections import CrossSection

AREA = 0.01  # m^2
LENGTH = 2.0  # m
APPLIED_FORCE = 1000.0  # N
DISPLACEMENT_TOLERANCE = 1e-9  # relative tolerance for engineering validation


def main() -> None:
    """Build, solve, and report results for a single-bar model."""
    steel = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    section = CrossSection(area=AREA)

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=LENGTH, y=0.0, z=0.0)
    bar = BarElement(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)

    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    mesh.add_element(bar)

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(node_id=node_1.id, dof=0, value=0.0))
    analysis.add_load(NodalLoad(node_id=node_2.id, dof=0, value=APPLIED_FORCE))
    result = analysis.solve()

    displacement = result.displacement(node_2.id)
    reaction = result.reaction(node_1.id)
    strain = result.element_strain(bar.id)
    stress = result.element_stress(bar.id)
    axial_force = result.element_axial_force(bar.id)

    analytical_displacement = APPLIED_FORCE * LENGTH / (steel.youngs_modulus * AREA)
    relative_error = abs(displacement - analytical_displacement) / analytical_displacement

    equilibrium_ok = abs(APPLIED_FORCE + reaction) < 1e-6
    validation_ok = relative_error < DISPLACEMENT_TOLERANCE

    print_summary(
        material=steel,
        area=AREA,
        applied_force=APPLIED_FORCE,
        displacement=displacement,
        reaction=reaction,
        strain=strain,
        stress=stress,
        axial_force=axial_force,
        analytical_displacement=analytical_displacement,
        relative_error=relative_error,
        equilibrium_ok=equilibrium_ok,
        validation_ok=validation_ok,
    )


def print_summary(
    material: Material,
    area: float,
    applied_force: float,
    displacement: float,
    reaction: float,
    strain: float,
    stress: float,
    axial_force: float,
    analytical_displacement: float,
    relative_error: float,
    equilibrium_ok: bool,
    validation_ok: bool,
) -> None:
    """Print a human-readable summary of the single-bar analysis results."""
    print("Finite Element Toolkit")
    print("Version 3 -- 1D Bar Analysis")
    print("=" * 40)

    print(f"\nMaterial:\n    {material.name}")
    print(f"\nYoung's modulus:\n    {material.youngs_modulus / 1e9:.0f} GPa")
    print(f"\nCross-sectional area:\n    {area} m^2")
    print(f"\nApplied load:\n    {applied_force:.0f} N")

    print(f"\nDisplacement:\n    {displacement:.6e} m")
    print(f"\nReaction:\n    {reaction:.6e} N")
    print(f"\nElement strain:\n    {strain:.6e}")
    print(f"\nElement stress:\n    {stress:.6e} Pa")
    print(f"\nElement axial force:\n    {axial_force:.6e} N")

    print(f"\nAnalytical displacement:\n    {analytical_displacement:.6e} m")
    print(f"\nRelative error:\n    {relative_error:.3e}")

    print(f"\nGlobal equilibrium:\n    {'PASS' if equilibrium_ok else 'FAIL'}")
    print(f"\nEngineering validation:\n    {'PASS' if validation_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
