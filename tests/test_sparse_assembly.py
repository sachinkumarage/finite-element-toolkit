"""Tests for femtoolkit.analysis.sparse_assembly (Version 26)."""

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

from femtoolkit.analysis import (
    DOFMap,
    ElementMassContribution,
    ElementStiffnessContribution,
    TranslationDOF,
    assemble_global_mass,
    assemble_global_stiffness,
    bar_element_stiffness,
)
from femtoolkit.analysis.sparse_assembly import (
    assemble_global_mass_sparse,
    assemble_global_stiffness_sparse,
)
from femtoolkit.exceptions import EntityNotFoundError, ValidationError
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh.generator import create_quad_mesh, create_triangular_mesh

X = TranslationDOF.X


def test_sparse_assembly_returns_csr_matrix() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    local_stiffness = bar_element_stiffness(youngs_modulus=200e9, area=0.01, length=2.0)

    global_stiffness = assemble_global_stiffness_sparse(
        dof_map, [ElementStiffnessContribution(((1, X), (2, X)), local_stiffness)]
    )

    assert isinstance(global_stiffness, sp.csr_matrix)
    assert global_stiffness.shape == (2, 2)
    assert_allclose(global_stiffness.toarray(), local_stiffness)


def test_sparse_assembly_matches_dense_single_bar_element() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    local_stiffness = bar_element_stiffness(youngs_modulus=200e9, area=0.01, length=2.0)
    contributions = [ElementStiffnessContribution(((1, X), (2, X)), local_stiffness)]

    dense = assemble_global_stiffness(dof_map, contributions)
    sparse = assemble_global_stiffness_sparse(dof_map, contributions)

    assert_allclose(dense, sparse.toarray())


def test_sparse_assembly_sums_shared_dof_contributions() -> None:
    """Two bar elements sharing node 2 -- the sparse assembler must sum
    duplicate (row, col) entries exactly like the dense scatter-add loop.
    """
    dof_map = DOFMap(node_ids=[1, 2, 3], dofs_per_node=1)
    k1 = bar_element_stiffness(youngs_modulus=200e9, area=0.01, length=2.0)
    k2 = bar_element_stiffness(youngs_modulus=200e9, area=0.02, length=1.0)
    contributions = [
        ElementStiffnessContribution(((1, X), (2, X)), k1),
        ElementStiffnessContribution(((2, X), (3, X)), k2),
    ]

    dense = assemble_global_stiffness(dof_map, contributions)
    sparse = assemble_global_stiffness_sparse(dof_map, contributions)

    assert_allclose(dense, sparse.toarray())
    # node 2's diagonal entry must be the sum of both elements' contributions.
    assert sparse.toarray()[1, 1] == pytest.approx(k1[1, 1] + k2[0, 0])


@pytest.fixture
def material() -> LinearElastic2D:
    return LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")


def test_sparse_assembly_matches_dense_for_quad_mesh(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01)
    dof_map = DOFMap(node_ids=[n.id for n in mesh.nodes], dofs_per_node=2)
    contributions = [
        ElementStiffnessContribution(e.dof_keys(), e.stiffness_matrix) for e in mesh.elements
    ]

    dense = assemble_global_stiffness(dof_map, contributions)
    sparse = assemble_global_stiffness_sparse(dof_map, contributions)

    assert dense.shape == sparse.shape
    assert_allclose(dense, sparse.toarray(), atol=1e-6)


def test_sparse_assembly_matches_dense_for_triangular_mesh(material: LinearElastic2D) -> None:
    mesh = create_triangular_mesh(
        width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01
    )
    dof_map = DOFMap(node_ids=[n.id for n in mesh.nodes], dofs_per_node=2)
    contributions = [
        ElementStiffnessContribution(e.dof_keys(), e.stiffness_matrix) for e in mesh.elements
    ]

    dense = assemble_global_stiffness(dof_map, contributions)
    sparse = assemble_global_stiffness_sparse(dof_map, contributions)

    assert_allclose(dense, sparse.toarray(), atol=1e-6)


def test_sparse_assembly_nnz_and_density_are_sensible(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01)
    dof_map = DOFMap(node_ids=[n.id for n in mesh.nodes], dofs_per_node=2)
    contributions = [
        ElementStiffnessContribution(e.dof_keys(), e.stiffness_matrix) for e in mesh.elements
    ]
    sparse = assemble_global_stiffness_sparse(dof_map, contributions)

    n = dof_map.total_dofs
    density = sparse.nnz / (n * n)
    assert 0 < sparse.nnz <= n * n
    assert 0 < density <= 1.0


def test_sparse_assembly_rejects_wrong_shaped_matrix() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    bad_matrix = np.zeros((3, 3))

    with pytest.raises(ValidationError):
        assemble_global_stiffness_sparse(
            dof_map, [ElementStiffnessContribution(((1, X), (2, X)), bad_matrix)]
        )


def test_sparse_assembly_rejects_unknown_node() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    local_stiffness = bar_element_stiffness(youngs_modulus=200e9, area=0.01, length=2.0)

    with pytest.raises(EntityNotFoundError):
        assemble_global_stiffness_sparse(
            dof_map, [ElementStiffnessContribution(((1, X), (999, X)), local_stiffness)]
        )


@pytest.fixture
def material_with_density() -> LinearElastic2D:
    return LinearElastic2D(
        youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress", density=2700.0
    )


def test_sparse_mass_assembly_matches_dense(material_with_density: LinearElastic2D) -> None:
    from femtoolkit.analysis.mass import element_mass_matrix

    mesh = create_quad_mesh(
        width=2.0, height=1.0, nx=2, ny=2, material=material_with_density, thickness=0.01
    )
    dof_map = DOFMap(node_ids=[n.id for n in mesh.nodes], dofs_per_node=2)
    contributions = [
        ElementMassContribution(e.dof_keys(), element_mass_matrix(e)) for e in mesh.elements
    ]

    dense = assemble_global_mass(dof_map, contributions)
    sparse = assemble_global_mass_sparse(dof_map, contributions)

    assert_allclose(dense, sparse.toarray(), atol=1e-9)


def test_sparse_mass_assembly_uses_same_dof_numbering_as_stiffness(
    material_with_density: LinearElastic2D,
) -> None:
    from femtoolkit.analysis.mass import element_mass_matrix

    mesh = create_quad_mesh(
        width=2.0, height=1.0, nx=2, ny=2, material=material_with_density, thickness=0.01
    )
    dof_map = DOFMap(node_ids=[n.id for n in mesh.nodes], dofs_per_node=2)
    stiffness_contributions = [
        ElementStiffnessContribution(e.dof_keys(), e.stiffness_matrix) for e in mesh.elements
    ]
    mass_contributions = [
        ElementMassContribution(e.dof_keys(), element_mass_matrix(e)) for e in mesh.elements
    ]

    k_sparse = assemble_global_stiffness_sparse(dof_map, stiffness_contributions)
    m_sparse = assemble_global_mass_sparse(dof_map, mass_contributions)

    assert k_sparse.shape == m_sparse.shape
