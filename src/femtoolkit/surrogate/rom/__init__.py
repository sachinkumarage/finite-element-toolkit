"""Reduced-order modeling: Proper Orthogonal Decomposition (Version 35).

.. code-block:: text

    Full solution
          |
          v
    Snapshot matrix (femtoolkit.surrogate.rom.snapshots)
          |
          v
    SVD
          |
          v
    Reduced basis V (femtoolkit.surrogate.rom.pod.PODModel)
          |
          v
    Reduced coordinates q
          |
          v
    Reconstructed solution
"""

from __future__ import annotations

from femtoolkit.surrogate.rom.base import (
    DEFAULT_RECONSTRUCTION_EPSILON,
    ReconstructionError,
    ReducedOrderModel,
    RomValidationReport,
    compute_reconstruction_error,
)
from femtoolkit.surrogate.rom.pod import PODModel
from femtoolkit.surrogate.rom.snapshots import (
    FieldSnapshot,
    build_snapshot_matrix,
    collect_field_snapshots_from_runs,
)

__all__ = [
    "DEFAULT_RECONSTRUCTION_EPSILON",
    "FieldSnapshot",
    "PODModel",
    "ReconstructionError",
    "ReducedOrderModel",
    "RomValidationReport",
    "build_snapshot_matrix",
    "collect_field_snapshots_from_runs",
    "compute_reconstruction_error",
]
