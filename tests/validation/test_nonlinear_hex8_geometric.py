"""Validation: HEX8 driven through Newton-Raphson under geometric nonlinearity (spec section 27)."""

import pytest

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.materials import LinearElastic3D, SaintVenantKirchhoff3D
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


def _build_analysis(load_per_node: float) -> NonlinearAnalysis:
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)

    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    material = SaintVenantKirchhoff3D(youngs_modulus=200e9, poisson_ratio=0.3)
    settings = NonlinearSolverSettings(load_steps=10, tolerance=1e-9, max_iterations=40)
    analysis = NonlinearAnalysis(mesh, {hexa.id: material}, settings, geometric_nonlinearity=True)
    for node_id in _FIXED_FACE:
        for dof in (X, Y, Z):
            analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))
    for node_id in _LOADED_FACE:
        analysis.add_load(NodalLoad(node_id, X, load_per_node))
    return analysis


def test_converges_under_large_displacement() -> None:
    analysis = _build_analysis(load_per_node=1.5e7)
    result = analysis.solve()
    assert result.converged
    assert all(step.iterations >= 1 for step in result.step_results)
    assert all(step.iterations < 40 for step in result.step_results)


def test_gauss_points_all_report_states() -> None:
    analysis = _build_analysis(load_per_node=1.0e7)
    result = analysis.solve()
    assert result.converged
    for gauss_point in range(8):
        state = result.element_state(1, gauss_point=gauss_point)
        assert state.strain.shape == (6,)
        assert state.stress.shape == (6,)


def test_reaction_forces_balance_applied_load() -> None:
    load_per_node = 1.2e7
    analysis = _build_analysis(load_per_node=load_per_node)
    result = analysis.solve()
    assert result.converged

    reaction_x_total = sum(result.reaction(node_id, X) for node_id in _FIXED_FACE)
    total_applied = load_per_node * len(_LOADED_FACE)
    assert reaction_x_total == pytest.approx(-total_applied, rel=1e-6)


def test_uniaxial_tension_stress_matches_nominal_stress_at_moderate_load() -> None:
    """At moderate load (small strain regime), sigma ~ F/A should hold approximately.

    Uses a proper "roller" uniaxial-tension boundary condition (X=0 on the
    whole fixed face, with only the minimal extra Y/Z pins needed to remove
    rigid-body modes) rather than fully clamping the fixed face -- fully
    clamping Y and Z there would introduce a genuine (non-bug) Poisson-
    restraint stress concentration near that face, which a single coarse
    element cannot resolve accurately (a Saint-Venant end effect).
    """
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)

    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    material = SaintVenantKirchhoff3D(youngs_modulus=200e9, poisson_ratio=0.3)
    settings = NonlinearSolverSettings(load_steps=10, tolerance=1e-9, max_iterations=40)
    analysis = NonlinearAnalysis(mesh, {hexa.id: material}, settings, geometric_nonlinearity=True)
    for node_id in _FIXED_FACE:
        analysis.add_boundary_condition(BoundaryCondition(node_id, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Z, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, Z, 0.0))
    load_per_node = 2.0e6  # modest, near-linear regime
    for node_id in _LOADED_FACE:
        analysis.add_load(NodalLoad(node_id, X, load_per_node))

    result = analysis.solve()
    assert result.converged

    total_load = load_per_node * len(_LOADED_FACE)
    nominal_stress = total_load / 1.0  # unit cross-section area
    for gauss_point in range(8):
        stress = result.element_stress(1, gauss_point=gauss_point)
        assert stress[0] == pytest.approx(nominal_stress, rel=0.05)
