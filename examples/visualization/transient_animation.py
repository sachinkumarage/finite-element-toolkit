"""Example: transient temperature animation, Version 23.

Solves the same transient heating problem as
``examples/post_processing/transient_temperature_history.py`` and
animates the resulting temperature field over every stored time step
via :meth:`~femtoolkit.postprocessing.visualization_3d.viewer.FEAViewer.animate`,
recording a GIF -- "basic animation... Step 0 -> 1 -> 2 -> ... -> N",
not a full video-rendering system.
"""

from pathlib import Path

from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.postprocessing import from_thermal_transient
from femtoolkit.postprocessing.visualization_3d import FEAViewer, ViewerConfig
from femtoolkit.thermal import PrescribedTemperature, ThermalMaterial
from femtoolkit.thermal.thermal_analysis import TransientThermalAnalysis

OUTPUT_PATH = Path(__file__).parent / "transient_animation.gif"

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def main() -> None:
    """Solve a transient heating problem and animate its temperature field."""
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
    analysis = TransientThermalAnalysis(
        mesh, {hexa.id: thermal_material}, time_step=20.0, num_steps=25, initial_temperature=293.15
    )
    for node in nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(PrescribedTemperature(node.id, 500.0))
    result = analysis.solve()

    simulation = from_thermal_transient(result)

    print("Finite Element Toolkit")
    print("Version 23 -- Transient Temperature Animation")
    print("=" * 46)
    print(f"\nSteps: {simulation.num_steps}, final time: {simulation.times[-1]:.1f} s")

    config = ViewerConfig(window_title="Transient Temperature Animation", off_screen=True)
    viewer = FEAViewer(simulation, config)
    viewer.show_edges(True)
    viewer.set_camera("isometric")
    viewer.animate(field_name="temperature", filename=str(OUTPUT_PATH), fps=5.0)
    viewer.close()
    print(f"\nSaved transient temperature animation to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
