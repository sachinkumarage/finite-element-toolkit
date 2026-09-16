"""Example: 3D temperature contour visualization, Version 23.

Solves the toolkit's mandatory linear temperature field benchmark on a
HEX8 block (``T(x) = T0 + a*x``, see ``temperature_gradient.py`` in
``examples/post_processing``) and displays the resulting 3D temperature
contour through :class:`~femtoolkit.postprocessing.visualization_3d.viewer.FEAViewer`,
reusing the Version 22 thermal result adapter -- no temperature
calculation happens in this script or in the viewer.
"""

from pathlib import Path

from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.postprocessing import from_thermal_steady_state
from femtoolkit.postprocessing.visualization_3d import FEAViewer, ViewerConfig
from femtoolkit.thermal import PrescribedTemperature, SteadyStateThermalAnalysis, ThermalMaterial

T0 = 300.0  # K
A = 40.0  # K/m
OUTPUT_PATH = Path(__file__).parent / "temperature_3d.png"

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def main() -> None:
    """Solve a linear temperature field and render its 3D contour."""
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
    print("Version 23 -- 3D Temperature Contour")
    print("=" * 40)
    temperatures = [simulation.final_step.nodal_value("temperature", n.id) for n in nodes]
    print(f"\nTemperature range: {min(temperatures):.2f} K to {max(temperatures):.2f} K")

    config = ViewerConfig(
        window_title="3D Temperature Contour", colormap="inferno", off_screen=True
    )
    viewer = FEAViewer(simulation, config)
    viewer.set_scalar_field("temperature")
    viewer.show_edges(True)
    viewer.set_camera("isometric")
    viewer.display()
    viewer.screenshot(str(OUTPUT_PATH))
    viewer.close()
    print(f"\nSaved temperature contour screenshot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
