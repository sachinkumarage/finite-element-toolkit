"""Validation: modal-superposition displacement reconstruction against a direct solution.

The whole premise of modal superposition (see
:mod:`femtoolkit.analysis.modal_superposition`) is that ``u(t) = sum_i(
phi_i * q_i(t))``, with each ``q_i(t)`` integrated as an independent
single-DOF oscillator, reproduces the *same* physical response as
directly integrating the full coupled system -- exactly, when every
mode is retained (both are, in the end, solving the same linear
``M u'' + C u' + K u = F(t)`` with the same Newmark-beta scheme, merely
in a different, mathematically equivalent coordinate basis), and
approximately, with a truncated mode set. This module validates both
claims directly against
:class:`~femtoolkit.analysis.dynamic_analysis.DynamicAnalysis`'s direct
time-history solution on a small FEM model.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import (
    BoundaryCondition,
    RayleighDamping,
    SinusoidalLoad,
    TimeDependentNodalLoad,
    TranslationDOF,
)
from femtoolkit.analysis.dynamic_analysis import DynamicAnalysis
from femtoolkit.analysis.dynamic_system import build_dynamic_system
from femtoolkit.analysis.modal_superposition import modal_superposition
from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import create_quad_mesh

YOUNGS_MODULUS = 200e9
POISSON_RATIO = 0.3
DENSITY = 7850.0
THICKNESS = 0.01
WIDTH = 0.5
HEIGHT = 0.1
TIME_STEP = 2e-5
TOTAL_TIME = 0.008


@pytest.fixture
def mesh_and_domain():
    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS,
        poisson_ratio=POISSON_RATIO,
        formulation="plane_stress",
        density=DENSITY,
    )
    domain = Rectangle(width=WIDTH, height=HEIGHT)
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=6, ny=2, material=material, thickness=THICKNESS
    )
    return mesh, domain


@pytest.fixture
def boundary_conditions(mesh_and_domain):
    mesh, domain = mesh_and_domain
    conditions = []
    for node in mesh.nodes_on_boundary(domain.boundary("left")):
        conditions.append(BoundaryCondition(node.id, TranslationDOF.X, 0.0))
        conditions.append(BoundaryCondition(node.id, TranslationDOF.Y, 0.0))
    return conditions


@pytest.fixture
def tip_node(mesh_and_domain):
    mesh, _ = mesh_and_domain
    return max((n for n in mesh.nodes if n.x == WIDTH), key=lambda n: n.y)


def test_full_mode_set_matches_direct_solution_exactly(
    mesh_and_domain, boundary_conditions, tip_node
) -> None:
    mesh, _ = mesh_and_domain
    damping = RayleighDamping(alpha=15.0, beta=0.0001)
    system = build_dynamic_system(mesh, boundary_conditions, damping=damping)
    n_free = system.dof_map.total_dofs - len(boundary_conditions)
    load = TimeDependentNodalLoad(tip_node.id, TranslationDOF.Y, SinusoidalLoad(80.0, 800.0))

    modal_result = modal_superposition(
        system, modes=n_free, loads=[load], time_step=TIME_STEP, total_time=TOTAL_TIME
    )

    direct = DynamicAnalysis(mesh, damping=damping)
    for bc in boundary_conditions:
        direct.add_boundary_condition(bc)
    direct.add_time_dependent_load(load)
    direct_result = direct.solve(time_step=TIME_STEP, total_time=TOTAL_TIME)

    assert_allclose(
        modal_result.displacement_history, direct_result.displacement_history, atol=1e-11
    )
    assert_allclose(modal_result.velocity_history, direct_result.velocity_history, atol=1e-8)
    assert_allclose(
        modal_result.acceleration_history, direct_result.acceleration_history, atol=1e-4
    )


def test_truncated_mode_set_converges_toward_direct_solution(
    mesh_and_domain, boundary_conditions, tip_node
) -> None:
    """More retained modes should monotonically reduce the reconstruction
    error relative to the direct solution.
    """
    mesh, _ = mesh_and_domain
    system = build_dynamic_system(mesh, boundary_conditions)
    load = TimeDependentNodalLoad(tip_node.id, TranslationDOF.Y, SinusoidalLoad(80.0, 800.0))

    direct = DynamicAnalysis(mesh)
    for bc in boundary_conditions:
        direct.add_boundary_condition(bc)
    direct.add_time_dependent_load(load)
    direct_result = direct.solve(time_step=TIME_STEP, total_time=TOTAL_TIME)
    direct_uy = direct_result.displacement(tip_node.id, TranslationDOF.Y)

    errors = []
    for modes in (2, 5, 10):
        modal_result = modal_superposition(
            system, modes=modes, loads=[load], time_step=TIME_STEP, total_time=TOTAL_TIME
        )
        modal_uy = modal_result.displacement(tip_node.id, TranslationDOF.Y)
        errors.append(np.abs(modal_uy - direct_uy).max())

    assert errors[0] >= errors[1] >= errors[2]
    assert errors[2] / np.abs(direct_uy).max() < 0.01


def test_reaction_history_matches_direct_solution_with_full_modes(
    mesh_and_domain, boundary_conditions, tip_node
) -> None:
    mesh, _ = mesh_and_domain
    system = build_dynamic_system(mesh, boundary_conditions)
    n_free = system.dof_map.total_dofs - len(boundary_conditions)
    load = TimeDependentNodalLoad(tip_node.id, TranslationDOF.Y, SinusoidalLoad(80.0, 800.0))

    modal_result = modal_superposition(
        system, modes=n_free, loads=[load], time_step=TIME_STEP, total_time=TOTAL_TIME
    )

    direct = DynamicAnalysis(mesh)
    for bc in boundary_conditions:
        direct.add_boundary_condition(bc)
    direct.add_time_dependent_load(load)
    direct_result = direct.solve(time_step=TIME_STEP, total_time=TOTAL_TIME)

    assert_allclose(modal_result.reaction_history, direct_result.reaction_history, atol=1e-3)
