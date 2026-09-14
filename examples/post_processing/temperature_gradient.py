"""Example: temperature gradient extraction, Version 22.

Solves the toolkit's mandatory analytical benchmark, a linear
temperature field ``T(x) = T0 + a*x`` (see
``tests/validation/test_postprocessing_analytical.py``), and reports the
post-processed temperature gradient and its magnitude against the exact
closed-form answer, ``grad(T) = a``.
"""

from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.postprocessing import from_thermal_steady_state, with_derived_fields
from femtoolkit.thermal import PrescribedTemperature, SteadyStateThermalAnalysis, ThermalMaterial

T0 = 300.0  # K
A = 40.0  # K/m
CONDUCTIVITY = 25.0  # W/(m*K)

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def main() -> None:
    """Solve a linear temperature field and verify its post-processed gradient."""
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

    simulation = with_derived_fields(from_thermal_steady_state(result))
    gradient = simulation.final_step.element_value("temperature_gradient", hexa.id)
    magnitude = simulation.final_step.element_value("temperature_gradient_magnitude", hexa.id)

    print("Finite Element Toolkit")
    print("Version 22 -- Temperature Gradient Extraction")
    print("=" * 47)
    print(f"\nField: T(x) = {T0} + {A}*x")
    print(f"Expected gradient: [{A}, 0, 0] K/m")
    print(f"Computed gradient: {gradient} K/m")
    print(f"Expected magnitude: {A} K/m")
    print(f"Computed magnitude: {magnitude:.6f} K/m")


if __name__ == "__main__":
    main()
