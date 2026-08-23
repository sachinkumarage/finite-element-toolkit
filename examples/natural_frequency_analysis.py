"""Example: FEM natural frequency (modal) analysis, Version 11.

Demonstrates the full Version 11 modal workflow on a small, fixed-left
Q4 plate:

.. code-block:: text

    Geometry -> Mesh -> DynamicSystem (mass + stiffness + boundary
        conditions) -> generalized eigenvalue problem -> natural
        frequencies and mode shapes

Reuses :func:`~femtoolkit.mesh.generator.create_quad_mesh` (Version 8)
and :func:`~femtoolkit.analysis.dynamic_system.build_dynamic_system`
(Version 11, itself built from the unmodified Version 2 assembly
machinery) -- no new element or solver code is needed to go from a
static mesh to a modal analysis.
"""

from femtoolkit.analysis import BoundaryCondition, TranslationDOF
from femtoolkit.analysis.dynamic_system import build_dynamic_system
from femtoolkit.analysis.modal import natural_frequencies_of_system
from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, create_quad_mesh

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
DENSITY = 7850.0  # kg/m^3 (structural steel)
THICKNESS = 0.01  # m
WIDTH = 0.5  # m
HEIGHT = 0.1  # m
NX = 10
NY = 3
NUM_MODES = 4


def main() -> None:
    """Build a cantilevered Q4 plate and solve for its lowest natural frequencies."""
    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS,
        poisson_ratio=POISSON_RATIO,
        formulation="plane_stress",
        density=DENSITY,
    )
    domain = Rectangle(width=WIDTH, height=HEIGHT)
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=NX, ny=NY, material=material, thickness=THICKNESS
    )

    boundary_conditions = []
    for node in mesh.nodes_on_boundary(domain.boundary("left")):
        boundary_conditions.append(BoundaryCondition(node.id, TranslationDOF.X, 0.0))
        boundary_conditions.append(BoundaryCondition(node.id, TranslationDOF.Y, 0.0))

    system = build_dynamic_system(mesh, boundary_conditions)
    result = natural_frequencies_of_system(system, num_modes=NUM_MODES)

    print_summary(mesh, system, result)


def print_summary(mesh: Mesh, system, result) -> None:
    """Print mass/stiffness matrix info and the computed modal results."""
    print("Finite Element Toolkit")
    print("Version 11 -- FEM Natural Frequency Analysis")
    print("=" * 40)

    print(f"\nDomain:\n    {WIDTH} m x {HEIGHT} m x {THICKNESS} m, density = {DENSITY} kg/m^3")
    print(f"\nMesh:\n    {len(mesh.nodes)} nodes, {len(mesh.elements)} Q4 elements")
    print(f"\nGlobal mass matrix shape:      {system.mass.shape}")
    print(f"Global stiffness matrix shape: {system.stiffness.shape}")

    print(f"\nLowest {NUM_MODES} natural frequencies:")
    for i in range(NUM_MODES):
        rigid_body = " (rigid-body)" if result.is_rigid_body_mode[i] else ""
        print(f"    Mode {i + 1}: f = {result.frequencies[i]:.3f} Hz{rigid_body}")

    print("\nMode shape normalization: max |displacement component| = 1.0")
    print(f"Mode 1 max abs component: {abs(result.mode_shapes[:, 0]).max():.6f}")
    print(f"Any rigid-body modes reported: {bool(result.is_rigid_body_mode.any())}")
    print("(none expected: the left edge is fully fixed)")


if __name__ == "__main__":
    main()
