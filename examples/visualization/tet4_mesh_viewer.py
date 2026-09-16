"""Example: 3D mesh visualization of a TET4 element, Version 23.

Builds a single 4-node tetrahedral element and renders it through
:class:`~femtoolkit.postprocessing.visualization_3d.viewer.FEAViewer` --
the simplest possible demonstration of the viewer and the mesh
conversion layer it relies on, with no field data attached yet.
"""

from pathlib import Path

from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Mesh, Node, Tet4Element3D
from femtoolkit.postprocessing.result_model import MeshTopology, ResultStep, SimulationResult
from femtoolkit.postprocessing.visualization_3d import FEAViewer, ViewerConfig

OUTPUT_PATH = Path(__file__).parent / "tet4_mesh_viewer.png"

_COORDS = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]


def main() -> None:
    """Build a single TET4 element and render it with the interactive viewer."""
    material = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    tet = Tet4Element3D(id=1, nodes=nodes, material=material)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(tet)

    topology = MeshTopology.from_mesh(mesh)
    result = SimulationResult(topology, steps=(ResultStep(index=0, time=0.0),))

    print("Finite Element Toolkit")
    print("Version 23 -- TET4 Mesh Visualization")
    print("=" * 40)
    print(f"\nNodes: {len(topology.node_ids)}, elements: {len(topology.element_ids)}")

    config = ViewerConfig(window_title="TET4 Mesh Viewer", off_screen=True)
    viewer = FEAViewer(result, config)
    viewer.set_camera("isometric")
    viewer.display()
    viewer.screenshot(str(OUTPUT_PATH))
    viewer.close()
    print(f"\nSaved TET4 mesh screenshot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
