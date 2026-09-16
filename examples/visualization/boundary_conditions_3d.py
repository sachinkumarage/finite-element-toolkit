"""Example: 3D boundary-condition and load visualization, Version 23.

Solves the same HEX8 uniaxial-tension case as ``stress_3d.py`` and adds
simple visual markers for the fixed (constrained) face and the loaded
face on top of the von Mises stress contour, via
:func:`~femtoolkit.postprocessing.visualization_3d.node_markers` -- a
point cloud at the affected nodes, added as an extra mesh through
:meth:`~femtoolkit.postprocessing.visualization_3d.viewer.FEAViewer.add_mesh`.
"""

from pathlib import Path

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.postprocessing import from_static_linear, with_derived_fields
from femtoolkit.postprocessing.visualization_3d import FEAViewer, ViewerConfig, node_markers

FACE_LOAD = 5.0e4  # N
OUTPUT_PATH = Path(__file__).parent / "boundary_conditions_3d.png"

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]

_FIXED_FACE = (1, 4, 5, 8)
_LOADED_FACE = (2, 3, 6, 7)


def main() -> None:
    """Solve a HEX8 element under uniaxial tension and mark its constraints and loads."""
    material = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=material)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    analysis = StaticLinearAnalysis(mesh)
    for node_id in _FIXED_FACE:
        for dof in (TranslationDOF.X, TranslationDOF.Y, TranslationDOF.Z):
            analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))
    for node_id in _LOADED_FACE:
        analysis.add_load(NodalLoad(node_id, TranslationDOF.X, FACE_LOAD))
    result = analysis.solve()

    simulation = with_derived_fields(from_static_linear(result))

    print("Finite Element Toolkit")
    print("Version 23 -- 3D Boundary Condition and Load Visualization")
    print("=" * 60)
    print(f"\nFixed face nodes: {_FIXED_FACE}")
    print(f"Loaded face nodes: {_LOADED_FACE}")

    config = ViewerConfig(window_title="Boundary Conditions and Loads", off_screen=True)
    viewer = FEAViewer(simulation, config)
    viewer.set_scalar_field("von_mises_stress")
    viewer.show_edges(True)
    viewer.set_camera("isometric")
    viewer.display()

    fixed_markers = node_markers(simulation.topology, _FIXED_FACE)
    viewer.add_mesh(
        "fixed_supports", fixed_markers, color="blue", point_size=20,
        render_points_as_spheres=True,
    )

    loaded_markers = node_markers(simulation.topology, _LOADED_FACE)
    viewer.add_mesh(
        "applied_loads", loaded_markers, color="red", point_size=20,
        render_points_as_spheres=True,
    )

    viewer.screenshot(str(OUTPUT_PATH))
    viewer.close()
    print(f"\nSaved boundary condition screenshot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
