"""Example: solve a cantilever beam under a tip load with Version 5.

This script demonstrates the complete Version 5 structural analysis
workflow for a 2D frame element:

    Material -> Cross-section -> Nodes -> Frame element -> Mesh
        -> Boundary conditions -> Loads -> Analysis -> Solution -> Results
        -> Analytical validation

Model:

    Fixed                         Free
    |-----------------------------o
    Node 1          L = 2 m       Node 2
                                   Fy = -1000 N

    E = 200 GPa, A = 0.01 m^2, I = 8.333e-6 m^4
    Node 1: fixed (ux = uy = rz = 0)
    Node 2: Fy = -1000 N (downward tip load)

The computed tip displacement, tip rotation, and fixed-end reaction are
reported, and checked against the classical Euler-Bernoulli cantilever
solution:

    delta = P * L^3 / (3 * E * I)   (tip deflection)
    theta = P * L^2 / (2 * E * I)   (tip rotation)
"""

from femtoolkit.analysis import (
    BoundaryCondition,
    NodalLoad,
    RotationDOF,
    StaticLinearAnalysis,
    TranslationDOF,
)
from femtoolkit.materials import Material
from femtoolkit.mesh import FrameElement2D, Mesh, Node
from femtoolkit.sections import CrossSection

AREA = 0.01  # m^2
SECOND_MOMENT_OF_AREA = 8.333e-6  # m^4
LENGTH = 2.0  # m
APPLIED_LOAD = 1000.0  # N, downward at the tip
EQUILIBRIUM_TOLERANCE = 1e-6  # N and N*m
VALIDATION_TOLERANCE = 1e-6  # relative


def main() -> None:
    """Build, solve, and report results for a cantilever beam model."""
    steel = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    section = CrossSection(area=AREA, second_moment_of_area=SECOND_MOMENT_OF_AREA)

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=LENGTH, y=0.0, z=0.0)
    beam = FrameElement2D(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)

    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    mesh.add_element(beam)

    analysis = StaticLinearAnalysis(mesh)
    for dof in (TranslationDOF.X, TranslationDOF.Y, RotationDOF.RZ):
        analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=dof, value=0.0))
    analysis.add_load(NodalLoad(node_id=2, dof=TranslationDOF.Y, value=-APPLIED_LOAD))
    result = analysis.solve()

    tip_displacement = result.displacement(2, TranslationDOF.Y)
    tip_rotation = result.displacement(2, RotationDOF.RZ)
    rx, ry, mz = result.node_reaction(1)
    bending_moment = result.element_bending_moment(1)
    shear_force = result.element_shear_force(1)

    e, i = steel.youngs_modulus, SECOND_MOMENT_OF_AREA
    analytical_displacement = -APPLIED_LOAD * LENGTH**3 / (3 * e * i)
    analytical_rotation = -APPLIED_LOAD * LENGTH**2 / (2 * e * i)

    equilibrium_ok = (
        abs(rx) < EQUILIBRIUM_TOLERANCE
        and abs(ry - APPLIED_LOAD) < EQUILIBRIUM_TOLERANCE
        and abs(mz - APPLIED_LOAD * LENGTH) < EQUILIBRIUM_TOLERANCE
    )
    displacement_error = abs(tip_displacement - analytical_displacement) / abs(
        analytical_displacement
    )
    rotation_error = abs(tip_rotation - analytical_rotation) / abs(analytical_rotation)
    validation_ok = (
        displacement_error < VALIDATION_TOLERANCE and rotation_error < VALIDATION_TOLERANCE
    )

    print_summary(
        material=steel,
        area=AREA,
        second_moment_of_area=SECOND_MOMENT_OF_AREA,
        applied_load=APPLIED_LOAD,
        tip_displacement=tip_displacement,
        analytical_displacement=analytical_displacement,
        tip_rotation=tip_rotation,
        analytical_rotation=analytical_rotation,
        reaction=(rx, ry, mz),
        bending_moment=bending_moment,
        shear_force=shear_force,
        equilibrium_ok=equilibrium_ok,
        validation_ok=validation_ok,
    )


def print_summary(
    material: Material,
    area: float,
    second_moment_of_area: float,
    applied_load: float,
    tip_displacement: float,
    analytical_displacement: float,
    tip_rotation: float,
    analytical_rotation: float,
    reaction: tuple[float, float, float],
    bending_moment: float,
    shear_force: float,
    equilibrium_ok: bool,
    validation_ok: bool,
) -> None:
    """Print a human-readable summary of the cantilever beam analysis results."""
    rx, ry, mz = reaction

    print("Finite Element Toolkit")
    print("Version 5 -- Cantilever Beam")
    print("=" * 40)

    print(f"\nBeam length:\n    {LENGTH} m")
    print(f"\nYoung's modulus:\n    {material.youngs_modulus / 1e9:.0f} GPa")
    print(f"\nArea:\n    {area} m^2")
    print(f"\nSecond moment of area:\n    {second_moment_of_area:.4e} m^4")
    print(f"\nApplied load:\n    {-applied_load:.0f} N (downward at the tip)")

    print(f"\nTip displacement:\n    {tip_displacement:.6e} m")
    print(f"\nAnalytical displacement:\n    {analytical_displacement:.6e} m")

    print(f"\nTip rotation:\n    {tip_rotation:.6e} rad")
    print(f"\nAnalytical rotation:\n    {analytical_rotation:.6e} rad")

    print(f"\nFixed-end reaction:\n    Rx = {rx:.6e} N, Ry = {ry:.6e} N, Mz = {mz:.6e} N*m")
    print(f"\nFixed-end bending moment:\n    {bending_moment:.6e} N*m")
    print(f"\nFixed-end shear force:\n    {shear_force:.6e} N")

    print(f"\nGlobal equilibrium:\n    {'PASS' if equilibrium_ok else 'FAIL'}")
    print(f"\nAnalytical validation:\n    {'PASS' if validation_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
