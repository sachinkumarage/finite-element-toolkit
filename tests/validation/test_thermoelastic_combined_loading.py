"""Validation: combined mechanical + thermal loading, and temperature-dependent properties
at the full Newton-Raphson solver level (spec sections 7, 15).
"""

import pytest

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.materials import LinearElastic3D, ThermoelasticMaterial3D
from femtoolkit.materials.thermal_properties import TemperatureDependentProperty
from femtoolkit.mesh import Hex8Element3D, Mesh, Node

X = TranslationDOF.X
Y = TranslationDOF.Y
Z = TranslationDOF.Z

_YOUNGS_MODULUS = 200e9
_POISSON_RATIO = 0.3
_ALPHA = 12e-6
_T_REF = 293.15

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]
_FIXED_FACE = (1, 4, 5, 8)
_LOADED_FACE = (2, 3, 6, 7)


def _build_mesh_and_hex() -> tuple[Mesh, Hex8Element3D]:
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    placeholder = LinearElastic3D(
        youngs_modulus=_YOUNGS_MODULUS, poisson_ratio=_POISSON_RATIO, density=7850.0
    )
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)
    return mesh, hexa


def test_combined_loading_matches_superposition_of_mechanical_and_thermal_alone() -> None:
    """For a linear thermoelastic material, combined loading must equal the exact sum
    of the mechanical-only and thermal-only solutions (superposition)."""
    base_material = ThermoelasticMaterial3D(
        youngs_modulus=_YOUNGS_MODULUS,
        poisson_ratio=_POISSON_RATIO,
        thermal_expansion_coefficient=_ALPHA,
        reference_temperature=_T_REF,
        density=7850.0,
    )
    temperature = 393.15
    load_per_node = 5.0e6

    def _solve(apply_load: bool, apply_heat: bool) -> float:
        mesh, hexa = _build_mesh_and_hex()
        material_temperature = temperature if apply_heat else _T_REF
        material = base_material.at_temperature(material_temperature)
        settings = NonlinearSolverSettings(load_steps=1, tolerance=1e-6, max_iterations=15)
        analysis = NonlinearAnalysis(
            mesh, {hexa.id: material}, settings, geometric_nonlinearity=False
        )
        for node_id in _FIXED_FACE:
            for dof in (X, Y, Z):
                analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))
        if apply_load:
            for node_id in _LOADED_FACE:
                analysis.add_load(NodalLoad(node_id, X, load_per_node))
        result = analysis.solve()
        assert result.converged
        return result.displacement(2, X)

    mechanical_only = _solve(apply_load=True, apply_heat=False)
    thermal_only = _solve(apply_load=False, apply_heat=True)
    combined = _solve(apply_load=True, apply_heat=True)

    assert combined == pytest.approx(mechanical_only + thermal_only, rel=1e-6)


def test_combined_loading_converges_in_two_iterations_per_step() -> None:
    """Linear thermoelasticity through Newton-Raphson must converge immediately: one
    iteration to find the exact linear correction, one more to confirm zero residual --
    exactly the behavior ElasticMaterialAdapter established for linear materials in
    Version 13 (verified directly against it above)."""
    base_material = ThermoelasticMaterial3D(
        youngs_modulus=_YOUNGS_MODULUS,
        poisson_ratio=_POISSON_RATIO,
        thermal_expansion_coefficient=_ALPHA,
        reference_temperature=_T_REF,
        density=7850.0,
    )
    mesh, hexa = _build_mesh_and_hex()
    material = base_material.at_temperature(393.15)
    settings = NonlinearSolverSettings(load_steps=5, tolerance=1e-6, max_iterations=15)
    analysis = NonlinearAnalysis(mesh, {hexa.id: material}, settings, geometric_nonlinearity=False)
    for node_id in _FIXED_FACE:
        for dof in (X, Y, Z):
            analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))
    for node_id in _LOADED_FACE:
        analysis.add_load(NodalLoad(node_id, X, 5.0e6))

    result = analysis.solve()
    assert result.converged
    assert all(step.iterations <= 2 for step in result.step_results)


def test_temperature_dependent_properties_change_the_solved_response() -> None:
    """A HEX8 solved at two different temperatures with a temperature-dependent E
    must give different stiffness/displacement, matching the tabulated softening.

    Each solve uses ``reference_temperature = temperature`` (``dT = 0``), so no
    thermal-expansion displacement confounds the comparison -- isolating purely
    the effect of evaluating ``E(T)`` at a different temperature.
    """
    youngs_modulus = TemperatureDependentProperty(
        temperatures=(293.15, 393.15), values=(200e9, 100e9)
    )

    def _solve_at(temperature: float) -> float:
        material_isolated = ThermoelasticMaterial3D(
            youngs_modulus=youngs_modulus,
            poisson_ratio=_POISSON_RATIO,
            thermal_expansion_coefficient=_ALPHA,
            reference_temperature=temperature,
            density=7850.0,
        )
        mesh, hexa = _build_mesh_and_hex()
        material = material_isolated.at_temperature(temperature)
        settings = NonlinearSolverSettings(load_steps=1, tolerance=1e-6, max_iterations=15)
        analysis = NonlinearAnalysis(
            mesh, {hexa.id: material}, settings, geometric_nonlinearity=False
        )
        for node_id in _FIXED_FACE:
            for dof in (X, Y, Z):
                analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))
        for node_id in _LOADED_FACE:
            analysis.add_load(NodalLoad(node_id, X, 1.0e6))
        result = analysis.solve()
        assert result.converged
        return result.displacement(2, X)

    displacement_cold = _solve_at(293.15)
    displacement_hot = _solve_at(393.15)

    # Half the stiffness (E halved) at the same mechanical load -> exactly double
    # the displacement.
    assert displacement_hot / displacement_cold == pytest.approx(2.0, rel=1e-6)
