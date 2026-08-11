"""Example: solve a cantilever beam under a pure end moment with Version 5.

Model:

    Fixed                    Free
    |-------------------------o
    Node 1        L = 2 m     Node 2
                               Mz = 500 N*m

    E = 200 GPa, I = 8.333e-6 m^4
    Node 1: fixed (ux = uy = rz = 0)
    Node 2: Mz = 500 N*m (applied nodal moment, no transverse load)

The computed tip rotation, tip displacement, and fixed-end reaction
moment are reported, and checked against the classical Euler-Bernoulli
solution for a cantilever under a pure end moment:

    theta = M * L / (E * I)         (tip rotation)
    delta = M * L^2 / (2 * E * I)   (tip deflection)
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
APPLIED_MOMENT = 500.0  # N*m, applied at the tip
EQUILIBRIUM_TOLERANCE = 1e-6  # N and N*m
VALIDATION_TOLERANCE = 1e-6  # relative


def main() -> None:
    """Build, solve, and report results for a cantilever-under-end-moment model."""
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
    analysis.add_load(NodalLoad(node_id=2, dof=RotationDOF.RZ, value=APPLIED_MOMENT))
    result = analysis.solve()

    tip_rotation = result.displacement(2, RotationDOF.RZ)
    tip_displacement = result.displacement(2, TranslationDOF.Y)
    rx, ry, mz = result.node_reaction(1)

    e, i = steel.youngs_modulus, SECOND_MOMENT_OF_AREA
    analytical_rotation = APPLIED_MOMENT * LENGTH / (e * i)
    analytical_displacement = APPLIED_MOMENT * LENGTH**2 / (2 * e * i)

    equilibrium_ok = (
        abs(rx) < EQUILIBRIUM_TOLERANCE
        and abs(ry) < EQUILIBRIUM_TOLERANCE
        and abs(mz + APPLIED_MOMENT) < EQUILIBRIUM_TOLERANCE
    )
    rotation_error = abs(tip_rotation - analytical_rotation) / abs(analytical_rotation)
    displacement_error = abs(tip_displacement - analytical_displacement) / abs(
        analytical_displacement
    )
    validation_ok = (
        rotation_error < VALIDATION_TOLERANCE and displacement_error < VALIDATION_TOLERANCE
    )

    print("Finite Element Toolkit")
    print("Version 5 -- Cantilever Beam Under a Pure End Moment")
    print("=" * 40)

    print(f"\nBeam length:\n    {LENGTH} m")
    print(f"\nYoung's modulus:\n    {steel.youngs_modulus / 1e9:.0f} GPa")
    print(f"\nSecond moment of area:\n    {SECOND_MOMENT_OF_AREA:.4e} m^4")
    print(f"\nApplied moment:\n    {APPLIED_MOMENT:.0f} N*m")

    print(f"\nTip rotation:\n    {tip_rotation:.6e} rad")
    print(f"\nAnalytical rotation:\n    {analytical_rotation:.6e} rad")

    print(f"\nTip displacement:\n    {tip_displacement:.6e} m")
    print(f"\nAnalytical displacement:\n    {analytical_displacement:.6e} m")

    print(f"\nFixed-end reaction:\n    Rx = {rx:.6e} N, Ry = {ry:.6e} N, Mz = {mz:.6e} N*m")

    print(f"\nGlobal equilibrium:\n    {'PASS' if equilibrium_ok else 'FAIL'}")
    print(f"\nAnalytical validation:\n    {'PASS' if validation_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
