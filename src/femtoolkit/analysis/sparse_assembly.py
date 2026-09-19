"""Global sparse matrix assembly (Version 26).

Dense assembly (:mod:`femtoolkit.analysis.assembly`, Version 2-11)
allocates a full ``(N, N)`` array up front and scatter-adds every
element contribution directly into it. For a mesh with many degrees of
freedom, that array is mostly zero: each element only connects nodes
that are physically close to each other, so a given global row has
non-zero entries only in the columns belonging to that node's own
neighboring elements.

.. code-block:: text

    Dense (all N^2 entries stored):
    [████████████████]
    [████████████████]
    [████████████████]
    [████████████████]

    Sparse (only nnz non-zero entries stored):
    [██░░░░░░░░░░░░██]
    [███░░░░░░░░░░░░░]
    [░░███░░░░░░░░░░░]
    [░░░░███░░░░░░░░░]

Storing only the non-zero entries turns an ``O(N^2)`` memory/compute
footprint into roughly ``O(N)`` for a typical FEA mesh (each row has a
bounded number of non-zero columns, independent of the mesh's overall
size), which is what makes solving substantially larger models
practical.

**This module assembles directly into sparse form** -- it never builds
a dense array and converts it, which spec section 6 explicitly warns
against ("avoid repeatedly converting a large sparse matrix between
dense and sparse formats"). It accumulates **COO** (coordinate format:
parallel ``row``/``col``/``value`` arrays, one entry per triplet)
triplets while scattering each element's local matrix, then converts
once to **CSR** (compressed sparse row) via
:meth:`scipy.sparse.coo_matrix.tocsr`. COO is the natural format for
*building* a sparse matrix (appending a triplet is O(1), and
`scipy` automatically **sums duplicate ``(row, col)`` triplets** during
the COO -> CSR conversion -- exactly the "elements sharing a DOF get
their stiffness contributions summed" behavior the dense scatter-add
loop provides, verified directly in this module's tests). CSR is the
natural format for *using* a sparse matrix afterward: fast row slicing
(needed to eliminate boundary-condition rows, see
:mod:`femtoolkit.solvers.base`) and fast matrix-vector products (needed
by every solver, direct or iterative).

Produces mathematically identical results to
:func:`~femtoolkit.analysis.assembly.assemble_global_stiffness`/
:func:`~femtoolkit.analysis.assembly.assemble_global_mass` for the same
inputs -- verified in ``tests/test_sparse_assembly.py`` by comparing
``K_dense`` against ``K_sparse.toarray()`` within floating-point
tolerance for several real element types.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp

from femtoolkit.analysis.assembly import (
    ElementMassContribution,
    ElementStiffnessContribution,
    _validate_contribution_shape,
)
from femtoolkit.analysis.dof import DOFMap


def _assemble_global_sparse_matrix(
    dof_map: DOFMap,
    contributions: Sequence[tuple[tuple[tuple[int, int], ...], np.ndarray]],
    matrix_label: str,
) -> sp.csr_matrix:
    """Accumulate COO triplets from element contributions and convert once to CSR.

    Shared by :func:`assemble_global_stiffness_sparse` and
    :func:`assemble_global_mass_sparse`; ``matrix_label`` only affects
    error messages, matching
    :func:`~femtoolkit.analysis.assembly._assemble_global_matrix`'s
    convention exactly.
    """
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []

    for dof_keys, local_matrix in contributions:
        _validate_contribution_shape(dof_keys, local_matrix, matrix_label)

        global_indices = [dof_map.global_index(node_id, dof) for node_id, dof in dof_keys]

        for local_row, global_row in enumerate(global_indices):
            for local_col, global_col in enumerate(global_indices):
                rows.append(global_row)
                cols.append(global_col)
                values.append(local_matrix[local_row, local_col])

    n = dof_map.total_dofs
    # coo_matrix sums duplicate (row, col) triplets automatically on tocsr(),
    # giving the same "shared-DOF contributions are summed" behavior as the
    # dense scatter-add loop.
    return sp.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()


def assemble_global_stiffness_sparse(
    dof_map: DOFMap,
    contributions: Sequence[ElementStiffnessContribution],
) -> sp.csr_matrix:
    """Assemble a global stiffness matrix directly in sparse (CSR) form.

    The sparse analogue of
    :func:`~femtoolkit.analysis.assembly.assemble_global_stiffness`,
    producing a mathematically equivalent result (``K_sparse.toarray()
    == K_dense`` within floating-point tolerance) from the exact same
    element contributions, without ever materializing a dense array.

    Args:
        dof_map: DOF map describing the global DOF numbering for all
            nodes involved.
        contributions: One
            :class:`~femtoolkit.analysis.assembly.ElementStiffnessContribution`
            per element in the model.

    Returns:
        The assembled global stiffness matrix as a
        :class:`scipy.sparse.csr_matrix`, of shape
        ``(dof_map.total_dofs, dof_map.total_dofs)``.

    Raises:
        ValidationError: If a contribution's stiffness matrix shape does
            not match its number of ``dof_keys``.
        EntityNotFoundError: If a contribution references a node ID that
            is not part of ``dof_map``.
    """
    return _assemble_global_sparse_matrix(dof_map, contributions, matrix_label="stiffness")


def assemble_global_mass_sparse(
    dof_map: DOFMap,
    contributions: Sequence[ElementMassContribution],
) -> sp.csr_matrix:
    """Assemble a global mass matrix directly in sparse (CSR) form.

    The sparse analogue of
    :func:`~femtoolkit.analysis.assembly.assemble_global_mass`. Provided
    for the same reason :func:`assemble_global_stiffness_sparse` is:
    keeping the door open for a future sparse dynamic-analysis solver
    (``M u'' + C u' + K u = F(t)``) without this version implementing
    one (see spec section 20) -- the mass matrix shares the exact same
    global DOF numbering as a stiffness matrix assembled from the same
    ``dof_map``, sparse or dense alike.

    Args:
        dof_map: DOF map describing the global DOF numbering for all
            nodes involved (the same one used for stiffness assembly).
        contributions: One
            :class:`~femtoolkit.analysis.assembly.ElementMassContribution`
            per element in the model.

    Returns:
        The assembled global mass matrix as a :class:`scipy.sparse.csr_matrix`.

    Raises:
        ValidationError: If a contribution's mass matrix shape does not
            match its number of ``dof_keys``.
        EntityNotFoundError: If a contribution references a node ID that
            is not part of ``dof_map``.
    """
    return _assemble_global_sparse_matrix(dof_map, contributions, matrix_label="mass")


__all__ = ["assemble_global_mass_sparse", "assemble_global_stiffness_sparse"]
