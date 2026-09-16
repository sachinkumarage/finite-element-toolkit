"""Example: 3D mesh visualization of a HEX8 element, Version 23.

Builds a single 8-node hexahedral (unit cube) element and renders it
through :class:`~femtoolkit.postprocessing.visualization_3d.viewer.FEAViewer`,
with mesh edges shown so the element's eight corner nodes and six faces
are clearly visible.
"""

from pathlib import Path

from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.postprocessing.result_model import MeshTopology, ResultStep, SimulationResult
from femtoolkit.postprocessing.visualization_3d import FEAViewer, ViewerConfig

OUTPUT_PATH = Path(__file__).parent / "hex8_mesh_viewer.png"

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def main() -> None:
    """Build a single HEX8 element and render it with the interactive viewer."""
    material = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=material)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    topology = MeshTopology.from_mesh(mesh)
    result = SimulationResult(topology, steps=(ResultStep(index=0, time=0.0),))

    print("Finite Element Toolkit")
    print("Version 23 -- HEX8 Mesh Visualization")
    print("=" * 40)
    print(f"\nNodes: {len(topology.node_ids)}, elements: {len(topology.element_ids)}")

    config = ViewerConfig(window_title="HEX8 Mesh Viewer", off_screen=True)
    viewer = FEAViewer(result, config)
    viewer.show_edges(True)
    viewer.set_camera("isometric")
    viewer.display()
    viewer.screenshot(str(OUTPUT_PATH))
    viewer.close()
    print(f"\nSaved HEX8 mesh screenshot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
