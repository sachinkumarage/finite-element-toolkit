"""Example: 3D von Mises stress contour visualization, Version 23.

Solves a HEX8 element under uniaxial tension (the Version 15 benchmark
from ``examples/hex8_linear_elastic.py``) and displays the resulting
von Mises equivalent stress as a 3D element-field contour, reusing the
Version 22 field calculator's ``von_mises_stress`` derived field --
no stress calculation happens in this script or in the viewer.
"""

from pathlib import Path

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.postprocessing import from_static_linear, with_derived_fields
from femtoolkit.postprocessing.visualization_3d import FEAViewer, ViewerConfig

FACE_LOAD = 5.0e4  # N, applied to each of the four x=1 face nodes
OUTPUT_PATH = Path(__file__).parent / "stress_3d.png"

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def main() -> None:
    """Solve a HEX8 element under uniaxial tension and render its von Mises stress."""
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
    von_mises = simulation.final_step.element_value("von_mises_stress", hexa.id)

    print("Finite Element Toolkit")
    print("Version 23 -- 3D Von Mises Stress Contour")
    print("=" * 42)
    print(f"\nVon Mises stress: {von_mises:.4e} Pa")

    config = ViewerConfig(window_title="3D Von Mises Stress", colormap="plasma", off_screen=True)
    viewer = FEAViewer(simulation, config)
    viewer.set_scalar_field("von_mises_stress")
    viewer.show_edges(True)
    viewer.set_camera("isometric")
    viewer.display()
    viewer.screenshot(str(OUTPUT_PATH))
    viewer.close()
    print(f"\nSaved von Mises stress screenshot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
