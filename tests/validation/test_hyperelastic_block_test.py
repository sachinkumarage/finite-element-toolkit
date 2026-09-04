"""Validation: single-HEX8 rubber block under extension/compression/shear (spec section 27).

Mirrors tests/validation/test_nonlinear_hex8_geometric.py's Newton-Raphson
patch test, but for the Version 17 hyperelastic materials specifically,
covering three canonical rubber-block load cases at genuinely large
strain (well beyond the small-strain regime a linear/St. Venant-Kirchhoff
material would be valid for).
"""

import pytest

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.materials import LinearElastic3D, MooneyRivlin3D, NeoHookean3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node

X = TranslationDOF.X
Y = TranslationDOF.Y
Z = TranslationDOF.Z

_COORDS = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.0),
]
_FIXED_FACE = (1, 4, 5, 8)
_LOADED_FACE = (2, 3, 6, 7)

_MATERIALS = [
    NeoHookean3D(youngs_modulus=5.0e6, poisson_ratio=0.45),
    MooneyRivlin3D(c10=0.4e6, c01=0.1e6, bulk_modulus=200e6),
]
_MATERIAL_IDS = ["neo_hookean", "mooney_rivlin"]


def _build_mesh_and_hex(placeholder: LinearElastic3D) -> tuple[Mesh, Hex8Element3D]:
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)
    return mesh, hexa


@pytest.mark.parametrize("material", _MATERIALS, ids=_MATERIAL_IDS)
def test_uniaxial_extension_converges_at_large_strain(material) -> None:
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    mesh, hexa = _build_mesh_and_hex(placeholder)

    settings = NonlinearSolverSettings(load_steps=20, tolerance=1e-8, max_iterations=40)
    analysis = NonlinearAnalysis(mesh, {hexa.id: material}, settings, geometric_nonlinearity=True)
    for node_id in _FIXED_FACE:
        for dof in (X, Y, Z):
            analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))
    load_per_node = 3.0e5
    for node_id in _LOADED_FACE:
        analysis.add_load(NodalLoad(node_id, X, load_per_node))

    result = analysis.solve()
    assert result.converged
    assert all(step.iterations < 40 for step in result.step_results)

    reaction_x_total = sum(result.reaction(node_id, X) for node_id in _FIXED_FACE)
    total_applied = load_per_node * len(_LOADED_FACE)
    assert reaction_x_total == pytest.approx(-total_applied, rel=1e-6)


@pytest.mark.parametrize("material", _MATERIALS, ids=_MATERIAL_IDS)
def test_uniaxial_compression_converges(material) -> None:
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    mesh, hexa = _build_mesh_and_hex(placeholder)

    settings = NonlinearSolverSettings(load_steps=20, tolerance=1e-8, max_iterations=40)
    analysis = NonlinearAnalysis(mesh, {hexa.id: material}, settings, geometric_nonlinearity=True)
    for node_id in _FIXED_FACE:
        for dof in (X, Y, Z):
            analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))
    load_per_node = -3.0e5
    for node_id in _LOADED_FACE:
        analysis.add_load(NodalLoad(node_id, X, load_per_node))

    result = analysis.solve()
    assert result.converged

    for node_id in _LOADED_FACE:
        displacement = result.node_displacement(node_id)
        assert displacement[0] < 0.0  # compressed inward


@pytest.mark.parametrize("material", _MATERIALS, ids=_MATERIAL_IDS)
def test_simple_shear_converges(material) -> None:
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    mesh, hexa = _build_mesh_and_hex(placeholder)

    settings = NonlinearSolverSettings(load_steps=20, tolerance=1e-8, max_iterations=40)
    analysis = NonlinearAnalysis(mesh, {hexa.id: material}, settings, geometric_nonlinearity=True)
    for node_id in (1, 4, 5, 8):
        for dof in (X, Y, Z):
            analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))
    # Shear the top-loaded face tangentially (Y direction) rather than pulling it away.
    load_per_node = 1.0e5
    for node_id in (2, 3, 6, 7):
        analysis.add_load(NodalLoad(node_id, Y, load_per_node))

    result = analysis.solve()
    assert result.converged
    for node_id in (2, 3, 6, 7):
        displacement = result.node_displacement(node_id)
        assert displacement[1] > 0.0  # sheared in the loaded direction


@pytest.mark.parametrize("material", _MATERIALS, ids=_MATERIAL_IDS)
def test_gauss_points_all_report_positive_jacobian_states(material) -> None:
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    mesh, hexa = _build_mesh_and_hex(placeholder)

    settings = NonlinearSolverSettings(load_steps=15, tolerance=1e-8, max_iterations=40)
    analysis = NonlinearAnalysis(mesh, {hexa.id: material}, settings, geometric_nonlinearity=True)
    for node_id in _FIXED_FACE:
        for dof in (X, Y, Z):
            analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))
    load_per_node = 2.0e5
    for node_id in _LOADED_FACE:
        analysis.add_load(NodalLoad(node_id, X, load_per_node))

    result = analysis.solve()
    assert result.converged
    for gauss_point in range(8):
        state = result.element_state(1, gauss_point=gauss_point)
        assert state.strain.shape == (6,)
        assert state.stress.shape == (6,)
