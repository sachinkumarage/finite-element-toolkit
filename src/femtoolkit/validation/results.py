"""``ValidationResult``: the outcome of comparing simulation to a reference dataset (Version 29).

Deliberately a separate type from
:class:`~femtoolkit.verification.cases.VerificationResult`, even though
the two share a similar shape -- keeping *validation* results in their
own type makes "this is validation, not verification" visible in a
function signature or a report section, matching spec section 10's
explicit instruction to distinguish the two throughout the API and not
just in prose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from femtoolkit.verification.status import VerificationStatus
from femtoolkit.verification.tolerance import Tolerance


@dataclass(frozen=True)
class ValidationResult:
    """The outcome of comparing a simulation result against a reference dataset.

    Attributes:
        dataset_name: The originating dataset's name.
        dataset_source: The originating dataset's source citation --
            always present, so a report never implies experimental
            agreement without saying where the reference data came from.
        quantity: The physical quantity being compared.
        simulation_values: The FEA values being validated, same shape
            as the dataset's ``values``.
        reference_values: The dataset's reference values.
        absolute_error: The L2 norm of the difference.
        relative_error: The relative L2 error.
        tolerance: The tolerance the comparison was checked against.
        status: The structured outcome.
        message: A short, human-readable summary.
        has_uncertainty: Whether the source dataset carried a reported
            uncertainty for its values (informational -- this
            comparison does not currently propagate it into the
            tolerance check; see ``docs/validation.md`` limitations).
    """

    dataset_name: str
    dataset_source: str
    quantity: str
    simulation_values: np.ndarray
    reference_values: np.ndarray
    absolute_error: float
    relative_error: float
    tolerance: Tolerance
    status: VerificationStatus
    message: str
    has_uncertainty: bool = False


__all__ = ["ValidationResult"]
