"""Validation: free thermal expansion of a HEX8/TET4 block (spec sections 5, 13, Case A).

Heating a body that is only minimally constrained (just enough to remove
rigid-body modes, not to resist expansion in any direction) must produce
genuine thermal *deformation* (matching the closed-form
``u = alpha*dT*X``) while developing approximately **zero** mechanical
stress -- the free thermal expansion case, one of this version's two
mandatory validation scenarios.
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


def _base_material() -> ThermoelasticMaterial3D:
    return ThermoelasticMaterial3D(
        youngs_modulus=_YOUNGS_MODULUS,
        poisson_ratio=_POISSON_RATIO,
        thermal_expansion_coefficient=_ALPHA,
        reference_temperature=_T_REF,
        density=7850.0,
    )


def test_hex8_free_expansion_gives_expected_displacement_and_near_zero_stress() -> None:
    coords = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(coords))
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

    # Minimal statically-determinate constraint: remove rigid body modes only.
    analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Z, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(2, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(2, Z, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, Z, 0.0))

    result = analysis.solve()
    assert result.converged

    delta_temperature = temperature - _T_REF
    expected_ux_node2 = _ALPHA * delta_temperature * 1.0
    assert result.displacement(2, X) == pytest.approx(expected_ux_node2, rel=1e-6)

    for gauss_point in range(8):
        state = result.element_state(1, gauss_point=gauss_point)
        assert np.abs(state.stress).max() < 1.0  # ~0 Pa: floating-point noise only


def test_tet4_free_expansion_gives_expected_displacement_and_near_zero_stress() -> None:
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

    analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Z, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(2, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(2, Z, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(3, Z, 0.0))

    result = analysis.solve()
    assert result.converged

    delta_temperature = temperature - _T_REF
    expected_ux_node2 = _ALPHA * delta_temperature * 1.0
    assert result.displacement(2, X) == pytest.approx(expected_ux_node2, rel=1e-6)

    state = result.element_state(1)
    assert np.abs(state.stress).max() < 1.0


def test_zero_temperature_change_gives_zero_displacement_and_zero_stress() -> None:
    coords = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(coords))
    placeholder = LinearElastic3D(
        youngs_modulus=_YOUNGS_MODULUS, poisson_ratio=_POISSON_RATIO, density=7850.0
    )
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)

    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    material = _base_material().at_temperature(_T_REF)
    settings = NonlinearSolverSettings(load_steps=1, tolerance=1e-6, max_iterations=15)
    analysis = NonlinearAnalysis(mesh, {hexa.id: material}, settings, geometric_nonlinearity=False)
    analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Z, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(2, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(2, Z, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, Z, 0.0))

    result = analysis.solve()
    assert result.converged
    assert result.displacement(2, X) == pytest.approx(0.0, abs=1e-9)
    for gauss_point in range(8):
        assert np.abs(result.element_state(1, gauss_point=gauss_point).stress).max() < 1.0
