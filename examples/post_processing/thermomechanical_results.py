"""Example: combined thermomechanical post-processing, Version 22.

Runs the sequential thermal -> mechanical workflow (Versions 19-21) and
merges both results into a single post-processing result via
:func:`~femtoolkit.postprocessing.merge_thermomechanical`, letting a
caller correlate temperature directly against the resulting
displacement/stress at the same nodes -- "temperature -> thermal
expansion -> stress/displacement" (spec section 10), without re-deriving
anything the thermal or mechanical solver did not already compute.
"""

from femtoolkit.analysis import BoundaryCondition, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.analysis.temperature_field import thermoelastic_materials_for_mesh
from femtoolkit.materials import LinearElastic3D, ThermoelasticMaterial3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.postprocessing import (
    PostProcessor,
    from_nonlinear,
    from_thermal_steady_state,
    merge_thermomechanical,
    with_derived_fields,
)
from femtoolkit.thermal import PrescribedTemperature, SteadyStateThermalAnalysis, ThermalMaterial

X = TranslationDOF.X
Y = TranslationDOF.Y
Z = TranslationDOF.Z

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def main() -> None:
    """Run the sequential thermal-then-mechanical workflow and correlate the results."""
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
    processor = PostProcessor(merged)

    print("Finite Element Toolkit")
    print("Version 22 -- Combined Thermomechanical Post-Processing")
    print("=" * 58)
    print(f"\nUniform applied temperature: {processor.mean('temperature'):.2f} K")

    print("\nPer-node correlation (temperature -> displacement):")
    for node in nodes:
        temperature = merged.final_step.nodal_value("temperature", node.id)
        displacement = merged.final_step.nodal_value("displacement", node.id)
        magnitude = (displacement**2).sum() ** 0.5
        print(f"    Node {node.id}: T={temperature:.2f} K, |u|={magnitude:.4e} m")

    max_von_mises = processor.maximum("von_mises_stress", kind="element")
    print(f"\nMaximum von Mises stress: {max_von_mises:.4e} Pa")
    print("(near-zero: this is a free-expansion case, so no thermal stress develops)")


if __name__ == "__main__":
    main()
