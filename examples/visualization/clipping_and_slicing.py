"""Example: 3D clipping, slicing, and thresholding, Version 23.

Solves the same linear temperature field as ``temperature_3d.py`` and
demonstrates the viewer's three internal-inspection tools --
:meth:`~femtoolkit.postprocessing.visualization_3d.viewer.FEAViewer.clip`,
:meth:`~femtoolkit.postprocessing.visualization_3d.viewer.FEAViewer.slice`,
and :meth:`~femtoolkit.postprocessing.visualization_3d.viewer.FEAViewer.threshold`
-- each producing its own screenshot of the same temperature field seen
a different way.
"""

from pathlib import Path

from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.postprocessing import from_thermal_steady_state
from femtoolkit.postprocessing.visualization_3d import FEAViewer, ViewerConfig
from femtoolkit.thermal import PrescribedTemperature, SteadyStateThermalAnalysis, ThermalMaterial

T0 = 300.0  # K
A = 40.0  # K/m
OUTPUT_CLIP = Path(__file__).parent / "clipping_and_slicing_clip.png"
OUTPUT_SLICE = Path(__file__).parent / "clipping_and_slicing_slice.png"
OUTPUT_THRESHOLD = Path(__file__).parent / "clipping_and_slicing_threshold.png"

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def main() -> None:
    """Solve a linear temperature field and demonstrate clip/slice/threshold inspection."""
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    thermal_material = ThermalMaterial(
        thermal_conductivity=25.0, density=7850.0, specific_heat=460.0
    )
    analysis = SteadyStateThermalAnalysis(mesh, {hexa.id: thermal_material})
    for node in nodes:
        analysis.add_boundary_condition(PrescribedTemperature(node.id, T0 + A * node.x))
    result = analysis.solve()

    simulation = from_thermal_steady_state(result)

    print("Finite Element Toolkit")
    print("Version 23 -- 3D Clipping, Slicing, and Thresholding")
    print("=" * 54)

    config = ViewerConfig(window_title="Clipping and Slicing", off_screen=True)
    viewer = FEAViewer(simulation, config)
    viewer.set_scalar_field("temperature")
    viewer.show_edges(True)
    viewer.set_camera("isometric")
    viewer.display()

    viewer.clip(normal=(1.0, 0.0, 0.0), origin=(0.5, 0.5, 0.5))
    viewer.screenshot(str(OUTPUT_CLIP))
    print(f"\nSaved clipped-mesh screenshot to {OUTPUT_CLIP}")

    viewer.display()
    viewer.slice(normal=(0.0, 0.0, 1.0), origin=(0.5, 0.5, 0.5))
    viewer.screenshot(str(OUTPUT_SLICE))
    print(f"Saved sliced-mesh screenshot to {OUTPUT_SLICE}")

    viewer.display()
    viewer.threshold("temperature", (T0 + 10.0, T0 + A))
    viewer.screenshot(str(OUTPUT_THRESHOLD))
    print(f"Saved thresholded-mesh screenshot to {OUTPUT_THRESHOLD}")

    viewer.close()


if __name__ == "__main__":
    main()
