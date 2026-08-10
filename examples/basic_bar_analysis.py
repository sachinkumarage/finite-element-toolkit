"""Example: solve a minimal 1D axial bar problem with the Version 2 foundation.

This script demonstrates the complete Version 2 mathematical workflow:

    Material -> Nodes -> Element -> element stiffness matrix
        -> global stiffness matrix -> boundary condition -> force vector
        -> linear system -> displacement

Model:

    Node 1 -------- Node 2
           Bar Element

    E = 200 GPa, A = 0.01 m^2, L = 2 m
    Node 1: fixed (u = 0)
    Node 2: F = 1000 N

The computed displacement at Node 2 is compared against the analytical
solution for a fixed-free axial bar, u = F * L / (E * A).

No stress or strain recovery is performed; that belongs to a future
version.
"""

import math

from femtoolkit.analysis import (
    BoundaryCondition,
    DOFMap,
    ElementStiffnessContribution,
    LinearSystem,
    NodalLoad,
    TranslationDOF,
    assemble_global_stiffness,
    bar_element_stiffness,
    build_force_vector,
    solve,
)
from femtoolkit.materials import Material
from femtoolkit.mesh import Element, Mesh, Node

AREA = 0.01  # m^2, cross-sectional area of the bar
APPLIED_FORCE = 1000.0  # N, axial force applied at Node 2


def main() -> None:
    """Build, solve, and report results for a two-node axial bar model."""
    steel = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)

    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    bar = Element(id=1, nodes=[node_1, node_2], material=steel)
    mesh.add_element(bar)

    length = math.dist((node_1.x, node_1.y, node_1.z), (node_2.x, node_2.y, node_2.z))

    dof_map = DOFMap(node_ids=[node.id for node in mesh.nodes], dofs_per_node=1)

    local_stiffness = bar_element_stiffness(
        youngs_modulus=steel.youngs_modulus, area=AREA, length=length
    )
    dof_keys = ((node_1.id, TranslationDOF.X), (node_2.id, TranslationDOF.X))
    global_stiffness = assemble_global_stiffness(
        dof_map,
        [ElementStiffnessContribution(dof_keys, local_stiffness)],
    )

    boundary_conditions = [BoundaryCondition(node_id=node_1.id, dof=0, value=0.0)]
    loads = [NodalLoad(node_id=node_2.id, dof=0, value=APPLIED_FORCE)]
    forces = build_force_vector(dof_map, loads)

    system = LinearSystem(
        dof_map=dof_map,
        stiffness=global_stiffness,
        forces=forces,
        boundary_conditions=boundary_conditions,
    )
    displacements = solve(system)

    computed_displacement = displacements[dof_map.global_index(node_2.id, 0)]
    analytical_displacement = APPLIED_FORCE * length / (steel.youngs_modulus * AREA)

    print_analysis_summary(
        material=steel,
        area=AREA,
        length=length,
        applied_force=APPLIED_FORCE,
        computed_displacement=computed_displacement,
        analytical_displacement=analytical_displacement,
    )


def print_analysis_summary(
    material: Material,
    area: float,
    length: float,
    applied_force: float,
    computed_displacement: float,
    analytical_displacement: float,
) -> None:
    """Print a human-readable summary of the bar analysis results.

    Args:
        material: Material used for the bar.
        area: Cross-sectional area of the bar, in square meters.
        length: Length of the bar, in meters.
        applied_force: Axial force applied at the free end, in newtons.
        computed_displacement: Displacement obtained from the FEA linear
            solve, in meters.
        analytical_displacement: Reference displacement from the
            analytical formula u = F * L / (E * A), in meters.
    """
    print("Finite Element Toolkit - Basic Bar Analysis")
    print("=" * 45)

    print(f"\nMaterial: {material.name}")
    print(f"Young's Modulus: {material.youngs_modulus / 1e9:.0f} GPa")
    print(f"Area: {area} m^2")
    print(f"Length: {length} m")
    print(f"Applied Load: {applied_force:.0f} N")

    print(f"\nComputed displacement:   {computed_displacement:.6e} m")
    print(f"Analytical displacement: {analytical_displacement:.6e} m")
    print(f"Difference:              {abs(computed_displacement - analytical_displacement):.3e} m")

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
