"""Example: 3D displacement visualization, Version 23.

Reuses the same HEX8 uniaxial-tension case as ``stress_3d.py`` and
displays displacement magnitude, then the X-component alone, as 3D
nodal-field contours -- demonstrating
:meth:`~femtoolkit.postprocessing.visualization_3d.viewer.FEAViewer.set_scalar_field`'s
``component`` argument for a vector field.
"""

from pathlib import Path

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.postprocessing import from_static_linear, with_derived_fields
from femtoolkit.postprocessing.visualization_3d import FEAViewer, ViewerConfig

FACE_LOAD = 5.0e4  # N
OUTPUT_MAGNITUDE = Path(__file__).parent / "displacement_3d_magnitude.png"
OUTPUT_X_COMPONENT = Path(__file__).parent / "displacement_3d_x.png"

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def main() -> None:
    """Solve a HEX8 element under uniaxial tension and render its displacement field."""
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
    max_magnitude = max(
        simulation.final_step.nodal_value("displacement_magnitude", n.id) for n in nodes
    )

    print("Finite Element Toolkit")
    print("Version 23 -- 3D Displacement Visualization")
    print("=" * 44)
    print(f"\nMaximum displacement magnitude: {max_magnitude:.6e} m")

    config = ViewerConfig(window_title="3D Displacement", off_screen=True)
    viewer = FEAViewer(simulation, config)

    viewer.set_scalar_field("displacement_magnitude")
    viewer.show_edges(True)
    viewer.set_camera("isometric")
    viewer.display()
    viewer.screenshot(str(OUTPUT_MAGNITUDE))
    print(f"\nSaved displacement magnitude screenshot to {OUTPUT_MAGNITUDE}")

    viewer.set_scalar_field("displacement", component=0)
    viewer.display()
    viewer.screenshot(str(OUTPUT_X_COMPONENT))
    print(f"Saved X-displacement screenshot to {OUTPUT_X_COMPONENT}")

    viewer.close()


if __name__ == "__main__":
    main()
