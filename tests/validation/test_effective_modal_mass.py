"""Validation: effective modal mass completeness for a physical FEM model.

For a complete eigenbasis of a constrained system's *free* DOFs, the sum
of effective modal mass over every mode exactly equals the total mass
participating in the requested direction -- restricted to the free
DOFs, since a fixed support cannot itself move and so contributes no
dynamic (participating) mass. This is an exact mathematical identity
(eigenbasis completeness: for mass-normalized modes,
``sum_i(phi_i @ phi_i^T) = M_free^-1``), not an approximation, so the
cumulative effective mass ratio must reach *exactly* ``1.0`` once every
free-DOF mode is included -- verified here on a small cantilevered Q4
plate (see :mod:`femtoolkit.analysis.modal`'s
``modal_analysis_of_system`` docstring for the derivation of why the
direction vector must be masked to zero at constrained DOFs for this to
hold).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, TranslationDOF
from femtoolkit.analysis.dynamic_system import build_dynamic_system
from femtoolkit.analysis.modal import modal_analysis_of_system
from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import create_quad_mesh

YOUNGS_MODULUS = 200e9
POISSON_RATIO = 0.3
DENSITY = 7850.0
THICKNESS = 0.01
WIDTH = 0.5
HEIGHT = 0.1


@pytest.fixture
def cantilever_system():
    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS,
        poisson_ratio=POISSON_RATIO,
        formulation="plane_stress",
        density=DENSITY,
    )
    domain = Rectangle(width=WIDTH, height=HEIGHT)
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=8, ny=3, material=material, thickness=THICKNESS
    )

    bcs = []
    for node in mesh.nodes_on_boundary(domain.boundary("left")):
        bcs.append(BoundaryCondition(node.id, TranslationDOF.X, 0.0))
        bcs.append(BoundaryCondition(node.id, TranslationDOF.Y, 0.0))

    return build_dynamic_system(mesh, bcs)


def test_cumulative_effective_mass_reaches_exactly_one_with_all_modes(cantilever_system) -> None:
    system = cantilever_system
    n_free = system.dof_map.total_dofs - len(system.boundary_conditions)

    result = modal_analysis_of_system(system, num_modes=n_free, direction="y")

    assert_allclose(result.cumulative_mass_ratio[-1], 1.0, rtol=1e-8)


def test_cumulative_effective_mass_is_monotonically_non_decreasing(cantilever_system) -> None:
    system = cantilever_system
    n_free = system.dof_map.total_dofs - len(system.boundary_conditions)

    result = modal_analysis_of_system(system, num_modes=n_free, direction="x")

    assert np.all(np.diff(result.cumulative_mass_ratio) >= -1e-12)


def test_partial_modes_cumulative_mass_is_strictly_less_than_full(cantilever_system) -> None:
    system = cantilever_system
    n_free = system.dof_map.total_dofs - len(system.boundary_conditions)

    partial = modal_analysis_of_system(system, num_modes=6, direction="y")
    full = modal_analysis_of_system(system, num_modes=n_free, direction="y")

    assert partial.cumulative_mass_ratio[-1] < full.cumulative_mass_ratio[-1] + 1e-9
    assert_allclose(full.cumulative_mass_ratio[-1], 1.0, rtol=1e-8)


def test_effective_mass_sum_equals_total_free_dof_mass_directly(cantilever_system) -> None:
    """Cross-check the completeness identity against a direct
    computation of the total participating mass from the reduced
    free-DOF mass matrix, independent of `modal_analysis_of_system`'s
    own internal bookkeeping.
    """
    system = cantilever_system
    n_free = system.dof_map.total_dofs - len(system.boundary_conditions)
    result = modal_analysis_of_system(system, num_modes=n_free, direction="y")

    constrained_indices = {
        system.dof_map.global_index(bc.node_id, bc.dof) for bc in system.boundary_conditions
    }
    total_dofs = system.dof_map.total_dofs
    free_indices = [i for i in range(total_dofs) if i not in constrained_indices]

    r_free_masked = np.zeros(total_dofs)
    for node_id in system.dof_map.node_ids:
        index = system.dof_map.global_index(node_id, TranslationDOF.Y)
        if index in free_indices:
            r_free_masked[index] = 1.0

    total_participating_mass = r_free_masked @ system.mass @ r_free_masked
    assert_allclose(result.effective_modal_mass.sum(), total_participating_mass, rtol=1e-8)
