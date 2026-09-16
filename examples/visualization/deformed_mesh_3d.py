"""Example: original vs. deformed 3D mesh comparison, Version 23.

Solves the same HEX8 uniaxial-tension case as ``stress_3d.py`` and
renders both the undeformed mesh and a visually scaled deformed mesh
side by side -- the true tip displacement on a unit cube under this
load is a small fraction of a millimeter, invisible at 1:1 scale, so a
large ``deformation_scale`` (purely cosmetic, see
:func:`~femtoolkit.postprocessing.field_calculator.deformed_coordinates`)
exaggerates it for visual clarity. The scale factor never changes any
physically computed result.
"""

from pathlib import Path

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.postprocessing import from_static_linear, with_derived_fields
from femtoolkit.postprocessing.visualization_3d import FEAViewer, ViewerConfig

FACE_LOAD = 5.0e4  # N
DEFORMATION_SCALE = 5000.0  # visual only
OUTPUT_UNDEFORMED = Path(__file__).parent / "deformed_mesh_3d_undeformed.png"
OUTPUT_DEFORMED = Path(__file__).parent / "deformed_mesh_3d_deformed.png"

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def main() -> None:
    """Solve a HEX8 element under uniaxial tension and compare original/deformed geometry."""
    material = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=material)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    analysis = StaticLinearAnalysis(mesh)
    fixed_face = (1, 4, 5, 8)
    loaded_face = (2, 3, 6, 7)
    for node_id in fixed_face:
        for dof in (TranslationDOF.X, TranslationDOF.Y, TranslationDOF.Z):
            analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))
    for node_id in loaded_face:
        analysis.add_load(NodalLoad(node_id, TranslationDOF.X, FACE_LOAD))
    result = analysis.solve()

    simulation = with_derived_fields(from_static_linear(result))
    max_displacement = max(
        simulation.final_step.nodal_value("displacement_magnitude", n.id) for n in nodes
    )

    print("Finite Element Toolkit")
    print("Version 23 -- Original vs. Deformed Mesh Comparison")
    print("=" * 53)
    print(f"\nTrue maximum displacement: {max_displacement:.6e} m")
    print(f"Visualization scale factor: {DEFORMATION_SCALE:g} (cosmetic only)")

    config = ViewerConfig(window_title="Deformed Mesh Comparison", off_screen=True)
    viewer = FEAViewer(simulation, config)
    viewer.set_scalar_field("von_mises_stress")
    viewer.show_edges(True)
    viewer.set_camera("isometric")

    viewer.show_undeformed()
    viewer.display()
    viewer.screenshot(str(OUTPUT_UNDEFORMED))
    print(f"\nSaved undeformed mesh screenshot to {OUTPUT_UNDEFORMED}")

    viewer.show_deformed()
    viewer.set_deformation_scale(DEFORMATION_SCALE)
    viewer.display()
    viewer.screenshot(str(OUTPUT_DEFORMED))
    print(f"Saved deformed mesh screenshot to {OUTPUT_DEFORMED}")

    viewer.close()


if __name__ == "__main__":
    main()
