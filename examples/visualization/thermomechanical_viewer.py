"""Example: combined thermomechanical 3D visualization, Version 23.

Runs the same sequential thermal -> mechanical workflow as
``examples/post_processing/thermomechanical_results.py`` and displays
the merged result's correlated temperature and von Mises stress fields
in the 3D viewer, one after another on the same deformed geometry --
reusing :func:`~femtoolkit.postprocessing.merge_thermomechanical`
entirely; the viewer performs no thermal or structural calculation.
"""

from pathlib import Path

from femtoolkit.analysis import BoundaryCondition, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.analysis.temperature_field import thermoelastic_materials_for_mesh
from femtoolkit.materials import LinearElastic3D, ThermoelasticMaterial3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.postprocessing import (
    from_nonlinear,
    from_thermal_steady_state,
    merge_thermomechanical,
    with_derived_fields,
)
from femtoolkit.postprocessing.visualization_3d import FEAViewer, ViewerConfig
from femtoolkit.thermal import PrescribedTemperature, SteadyStateThermalAnalysis, ThermalMaterial

X = TranslationDOF.X
Y = TranslationDOF.Y
Z = TranslationDOF.Z

OUTPUT_TEMPERATURE = Path(__file__).parent / "thermomechanical_viewer_temperature.png"
OUTPUT_STRESS = Path(__file__).parent / "thermomechanical_viewer_stress.png"

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def main() -> None:
    """Run the sequential thermal-then-mechanical workflow and visualize the merged result."""
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
    thermal_analysis = SteadyStateThermalAnalysis(mesh, {hexa.id: thermal_material})
    for node in nodes:
        thermal_analysis.add_boundary_condition(PrescribedTemperature(node.id, 450.0))
    thermal_result = thermal_analysis.solve()
    temperature_field = thermal_result.to_temperature_field()

    base_material = ThermoelasticMaterial3D(
        youngs_modulus=200e9,
        poisson_ratio=0.3,
        thermal_expansion_coefficient=12e-6,
        reference_temperature=293.15,
        density=7850.0,
    )
    materials = thermoelastic_materials_for_mesh(mesh, base_material, temperature_field)
    settings = NonlinearSolverSettings(load_steps=1, tolerance=1e-6, max_iterations=15)
    mechanical_analysis = NonlinearAnalysis(mesh, materials, settings, geometric_nonlinearity=False)
    mechanical_analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(1, Z, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(2, Y, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(2, Z, 0.0))
    mechanical_analysis.add_boundary_condition(BoundaryCondition(4, Z, 0.0))
    mechanical_result = mechanical_analysis.solve()

    thermal_simulation = from_thermal_steady_state(thermal_result)
    mechanical_simulation = with_derived_fields(from_nonlinear(mechanical_result, mesh))
    merged = merge_thermomechanical(thermal_simulation, mechanical_simulation)

    print("Finite Element Toolkit")
    print("Version 23 -- Combined Thermomechanical 3D Visualization")
    print("=" * 58)
    print(f"\nApplied temperature: {merged.final_step.nodal_value('temperature', 1):.2f} K")

    config = ViewerConfig(window_title="Thermomechanical Viewer", off_screen=True)
    viewer = FEAViewer(merged, config)
    viewer.show_edges(True)
    viewer.set_camera("isometric")

    viewer.set_scalar_field("temperature")
    viewer.display()
    viewer.screenshot(str(OUTPUT_TEMPERATURE))
    print(f"\nSaved temperature screenshot to {OUTPUT_TEMPERATURE}")

    viewer.set_scalar_field("von_mises_stress")
    viewer.show_deformed()
    viewer.set_deformation_scale(2000.0)
    viewer.display()
    viewer.screenshot(str(OUTPUT_STRESS))
    print(f"Saved deformed von Mises stress screenshot to {OUTPUT_STRESS}")

    viewer.close()


if __name__ == "__main__":
    main()
