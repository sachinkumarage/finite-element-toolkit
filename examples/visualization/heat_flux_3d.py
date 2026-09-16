"""Example: 3D heat-flux vector visualization, Version 23.

Solves the same linear temperature field as ``temperature_3d.py`` and
displays the resulting (uniform, ``-k*grad(T)``) heat-flux vector as a
single glyph arrow at the element centroid, via
:meth:`~femtoolkit.postprocessing.visualization_3d.viewer.FEAViewer.set_vector_field`
-- the heat flux itself comes entirely from the Version 22 thermal
result adapter, never recomputed here.
"""

from pathlib import Path

from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.postprocessing import from_thermal_steady_state
from femtoolkit.postprocessing.visualization_3d import FEAViewer, ViewerConfig
from femtoolkit.thermal import PrescribedTemperature, SteadyStateThermalAnalysis, ThermalMaterial

T0 = 300.0  # K
A = 40.0  # K/m
CONDUCTIVITY = 25.0  # W/(m*K)
OUTPUT_PATH = Path(__file__).parent / "heat_flux_3d.png"

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def main() -> None:
    """Solve a linear temperature field and render its 3D heat-flux vector."""
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    thermal_material = ThermalMaterial(
        thermal_conductivity=CONDUCTIVITY, density=7850.0, specific_heat=460.0
    )
    analysis = SteadyStateThermalAnalysis(mesh, {hexa.id: thermal_material})
    for node in nodes:
        analysis.add_boundary_condition(PrescribedTemperature(node.id, T0 + A * node.x))
    result = analysis.solve()

    simulation = from_thermal_steady_state(result)
    flux = simulation.final_step.element_value("heat_flux", hexa.id)

    print("Finite Element Toolkit")
    print("Version 23 -- 3D Heat-Flux Vector Visualization")
    print("=" * 48)
    expected = -CONDUCTIVITY * A
    print(f"\nElement heat flux: {flux} W/m^2 (expected [{expected}, 0, 0])")

    config = ViewerConfig(window_title="3D Heat-Flux Vectors", off_screen=True)
    viewer = FEAViewer(simulation, config)
    viewer.set_scalar_field("temperature")
    viewer.set_vector_field("heat_flux")
    viewer.show_edges(True)
    viewer.set_camera("isometric")
    viewer.display()
    viewer.screenshot(str(OUTPUT_PATH))
    viewer.close()
    print(f"\nSaved heat-flux vector screenshot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
