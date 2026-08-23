"""Validation Case: a physical FEM dynamic model -- a small fixed-left Q4 plate.

Builds a small cantilevered plate (a known material density, a fixed
left edge) entirely from existing, previously validated building blocks
-- :func:`~femtoolkit.mesh.generator.create_quad_mesh` (Version 8),
:func:`~femtoolkit.analysis.dynamic_system.build_dynamic_system`
(Version 11) -- and checks that the resulting mass matrix, stiffness
matrix, natural frequencies, and mode shapes are all finite and
physically reasonable:

* the assembled mass matrix's total (per-direction) mass matches
  ``density * area * thickness`` exactly (Version 11's mass-consistency
  requirement, re-verified here at the full analysis level rather than
  the isolated assembly level covered in ``tests/test_dynamic_system.py``);
* every natural frequency is finite, positive, and ascending;
* no rigid-body modes are reported (the model is properly constrained);
* every mode shape is normalized to a maximum absolute value of ``1.0``
  and is exactly zero at every constrained DOF;
* increasing mesh density leaves the lowest natural frequency
  approximately unchanged (a basic mesh-convergence sanity check, not a
  rigorous convergence study).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, TranslationDOF
from femtoolkit.analysis.dynamic_system import build_dynamic_system
from femtoolkit.analysis.modal import natural_frequencies_of_system
from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import create_quad_mesh

YOUNGS_MODULUS = 200e9
POISSON_RATIO = 0.3
DENSITY = 7850.0
THICKNESS = 0.01
WIDTH = 0.5
HEIGHT = 0.1


def _build_system(nx: int, ny: int):
    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS,
        poisson_ratio=POISSON_RATIO,
        formulation="plane_stress",
        density=DENSITY,
    )
    domain = Rectangle(width=WIDTH, height=HEIGHT)
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=nx, ny=ny, material=material, thickness=THICKNESS
    )

    bcs = []
    for node in mesh.nodes_on_boundary(domain.boundary("left")):
        bcs.append(BoundaryCondition(node.id, TranslationDOF.X, 0.0))
        bcs.append(BoundaryCondition(node.id, TranslationDOF.Y, 0.0))

    return build_dynamic_system(mesh, bcs), mesh


@pytest.fixture
def system_and_mesh():
    return _build_system(nx=6, ny=2)


def test_mass_matrix_total_matches_physical_mass(system_and_mesh) -> None:
    system, _ = system_and_mesh
    n = system.dof_map.total_dofs
    x_indices = [i for i in range(n) if i % 2 == 0]
    x_block = system.mass[np.ix_(x_indices, x_indices)]

    expected_mass = DENSITY * (WIDTH * HEIGHT) * THICKNESS
    assert_allclose(x_block.sum(), expected_mass)


def test_stiffness_matrix_is_symmetric(system_and_mesh) -> None:
    """Absolute tolerance matters here: entries near zero (e.g.
    unconnected DOF pairs) can show large *relative* floating-point
    noise from assembly summation order despite a tiny absolute
    difference, against stiffness entries of order 1e9.
    """
    system, _ = system_and_mesh
    assert_allclose(system.stiffness, system.stiffness.T, atol=1e-3)


def test_mass_matrix_is_symmetric(system_and_mesh) -> None:
    system, _ = system_and_mesh
    assert_allclose(system.mass, system.mass.T, atol=1e-12)


def test_natural_frequencies_are_finite_positive_ascending(system_and_mesh) -> None:
    system, _ = system_and_mesh
    result = natural_frequencies_of_system(system, num_modes=6)

    assert np.isfinite(result.frequencies).all()
    assert (result.frequencies > 0).all()
    assert np.all(np.diff(result.frequencies) >= -1e-9)


def test_no_rigid_body_modes_for_cantilevered_plate(system_and_mesh) -> None:
    system, _ = system_and_mesh
    result = natural_frequencies_of_system(system, num_modes=6)
    assert not result.is_rigid_body_mode.any()


def test_mode_shapes_normalized_and_zero_at_constraints(system_and_mesh) -> None:
    system, _ = system_and_mesh
    result = natural_frequencies_of_system(system, num_modes=4)

    for mode_index in range(4):
        assert_allclose(np.max(np.abs(result.mode_shapes[:, mode_index])), 1.0)

    for bc in system.boundary_conditions:
        index = system.dof_map.global_index(bc.node_id, bc.dof)
        assert_allclose(result.mode_shapes[index, :], np.zeros(4), atol=1e-12)


def test_first_bending_frequency_is_stable_under_mesh_refinement() -> None:
    """A basic mesh-convergence sanity check (not a formal convergence
    study): the fundamental frequency of a moderately refined and a
    finer mesh of the same physical plate should agree reasonably well.

    A *very* coarse mesh (e.g. a single row of elements through this
    thin, bending-dominated plate) is deliberately excluded here: fully
    integrated Q4 elements are well known to over-stiffen ("shear lock")
    in coarse bending-dominated meshes, so the very-coarse-mesh frequency
    converges to the fine-mesh value from *above*, monotonically, as the
    mesh is refined -- a real, expected numerical characteristic of this
    element formulation, not a bug (see Version 7's documented
    limitations). Comparing two already-reasonable mesh densities avoids
    that regime while still checking basic convergence behavior.
    """
    coarse_system, _ = _build_system(nx=10, ny=3)
    fine_system, _ = _build_system(nx=30, ny=8)

    coarse_result = natural_frequencies_of_system(coarse_system, num_modes=1)
    fine_result = natural_frequencies_of_system(fine_system, num_modes=1)

    assert_allclose(coarse_result.frequencies[0], fine_result.frequencies[0], rtol=0.1)


def test_lumped_and_consistent_mass_give_similar_but_not_identical_frequencies() -> None:
    """Lumped mass is a documented approximation of consistent mass (see
    femtoolkit.continuum.mass): natural frequencies from the two should
    be close but not require exact agreement.
    """
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
    bcs = [
        BoundaryCondition(node.id, dof, 0.0)
        for node in mesh.nodes_on_boundary(domain.boundary("left"))
        for dof in (TranslationDOF.X, TranslationDOF.Y)
    ]

    consistent_system = build_dynamic_system(mesh, bcs, mass_matrix_type="consistent")
    lumped_system = build_dynamic_system(mesh, bcs, mass_matrix_type="lumped")

    consistent_result = natural_frequencies_of_system(consistent_system, num_modes=1)
    lumped_result = natural_frequencies_of_system(lumped_system, num_modes=1)

    assert_allclose(consistent_result.frequencies[0], lumped_result.frequencies[0], rtol=0.2)
    assert consistent_result.frequencies[0] != lumped_result.frequencies[0]
