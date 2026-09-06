"""Validation: fully constrained thermal expansion of a HEX8/TET4 block (spec section 13, Case B).

Heating (or cooling) a body whose every DOF is fixed prevents any thermal
displacement at all: the total strain stays exactly zero, so the entire
thermal eigenstrain becomes mechanical strain, developing genuine
**thermal stress** -- the second of this version's two mandatory
validation scenarios. The closed-form fully-triaxially-restrained
thermal stress, ``sigma = -E*alpha*dT / (1 - 2*v)``, is checked exactly.
"""

import numpy as np
import pytest

from femtoolkit.analysis import BoundaryCondition, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.materials import LinearElastic3D, ThermoelasticMaterial3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node, Tet4Element3D

X = TranslationDOF.X
Y = TranslationDOF.Y
Z = TranslationDOF.Z

_YOUNGS_MODULUS = 200e9
_POISSON_RATIO = 0.3
_ALPHA = 12e-6
_T_REF = 293.15

_HEX8_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]


def _base_material() -> ThermoelasticMaterial3D:
    return ThermoelasticMaterial3D(
        youngs_modulus=_YOUNGS_MODULUS,
        poisson_ratio=_POISSON_RATIO,
        thermal_expansion_coefficient=_ALPHA,
        reference_temperature=_T_REF,
        density=7850.0,
    )


def _expected_constrained_normal_stress(delta_temperature: float) -> float:
    return -_YOUNGS_MODULUS * _ALPHA * delta_temperature / (1.0 - 2.0 * _POISSON_RATIO)


def test_hex8_fully_constrained_heating_matches_closed_form_thermal_stress() -> None:
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_HEX8_COORDS))
    placeholder = LinearElastic3D(
        youngs_modulus=_YOUNGS_MODULUS, poisson_ratio=_POISSON_RATIO, density=7850.0
    )
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)

    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    temperature = 393.15
    material = _base_material().at_temperature(temperature)
    settings = NonlinearSolverSettings(load_steps=1, tolerance=1e-6, max_iterations=15)
    analysis = NonlinearAnalysis(mesh, {hexa.id: material}, settings, geometric_nonlinearity=False)
    for node in nodes:
        for dof in (X, Y, Z):
            analysis.add_boundary_condition(BoundaryCondition(node.id, dof, 0.0))

    result = analysis.solve()
    assert result.converged

    expected_stress = _expected_constrained_normal_stress(temperature - _T_REF)
    for gauss_point in range(8):
        state = result.element_state(1, gauss_point=gauss_point)
        assert np.allclose(state.strain, np.zeros(6), atol=1e-12)
        assert state.stress[0] == pytest.approx(expected_stress, rel=1e-8)
        assert state.stress[1] == pytest.approx(expected_stress, rel=1e-8)
        assert state.stress[2] == pytest.approx(expected_stress, rel=1e-8)
        assert np.allclose(state.stress[3:], 0.0, atol=1.0)
    assert expected_stress < 0.0  # compressive under restrained heating


def test_hex8_fully_constrained_cooling_gives_tensile_thermal_stress() -> None:
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_HEX8_COORDS))
    placeholder = LinearElastic3D(
        youngs_modulus=_YOUNGS_MODULUS, poisson_ratio=_POISSON_RATIO, density=7850.0
    )
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)

    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    temperature = 193.15
    material = _base_material().at_temperature(temperature)
    settings = NonlinearSolverSettings(load_steps=1, tolerance=1e-6, max_iterations=15)
    analysis = NonlinearAnalysis(mesh, {hexa.id: material}, settings, geometric_nonlinearity=False)
    for node in nodes:
        for dof in (X, Y, Z):
            analysis.add_boundary_condition(BoundaryCondition(node.id, dof, 0.0))

    result = analysis.solve()
    assert result.converged

    expected_stress = _expected_constrained_normal_stress(temperature - _T_REF)
    state = result.element_state(1, gauss_point=0)
    assert state.stress[0] == pytest.approx(expected_stress, rel=1e-8)
    assert expected_stress > 0.0  # tensile under restrained cooling


def test_tet4_fully_constrained_heating_matches_closed_form_thermal_stress() -> None:
    nodes = (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=1.0, y=0.0, z=0.0),
        Node(id=3, x=0.0, y=1.0, z=0.0),
        Node(id=4, x=0.0, y=0.0, z=1.0),
    )
    placeholder = LinearElastic3D(
        youngs_modulus=_YOUNGS_MODULUS, poisson_ratio=_POISSON_RATIO, density=7850.0
    )
    tet = Tet4Element3D(id=1, nodes=nodes, material=placeholder)

    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(tet)

    temperature = 373.15
    material = _base_material().at_temperature(temperature)
    settings = NonlinearSolverSettings(load_steps=1, tolerance=1e-6, max_iterations=15)
    analysis = NonlinearAnalysis(mesh, {tet.id: material}, settings, geometric_nonlinearity=False)
    for node in nodes:
        for dof in (X, Y, Z):
            analysis.add_boundary_condition(BoundaryCondition(node.id, dof, 0.0))

    result = analysis.solve()
    assert result.converged

    expected_stress = _expected_constrained_normal_stress(temperature - _T_REF)
    state = result.element_state(1)
    assert state.stress[0] == pytest.approx(expected_stress, rel=1e-8)


def test_reaction_forces_balance_the_thermal_internal_stress() -> None:
    """With no external load, reactions must exactly balance F_int (R = F_int - F_ext = F_int)."""
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_HEX8_COORDS))
    placeholder = LinearElastic3D(
        youngs_modulus=_YOUNGS_MODULUS, poisson_ratio=_POISSON_RATIO, density=7850.0
    )
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)

    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    material = _base_material().at_temperature(393.15)
    settings = NonlinearSolverSettings(load_steps=1, tolerance=1e-6, max_iterations=15)
    analysis = NonlinearAnalysis(mesh, {hexa.id: material}, settings, geometric_nonlinearity=False)
    for node in nodes:
        for dof in (X, Y, Z):
            analysis.add_boundary_condition(BoundaryCondition(node.id, dof, 0.0))

    result = analysis.solve()
    assert result.converged
    # Sum of reactions in X over the whole (self-equilibrated) block must vanish.
    total_reaction_x = sum(result.reaction(node.id, X) for node in nodes)
    assert total_reaction_x == pytest.approx(0.0, abs=1.0)
