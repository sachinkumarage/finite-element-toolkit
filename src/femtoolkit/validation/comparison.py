"""Compares simulation results against a reference dataset (Version 29).

The one function here, :func:`compare_to_reference_dataset`, is the
entire *validation* comparison surface of this toolkit: it interpolates
nothing, assumes nothing about how ``simulation_values`` was produced,
and never claims agreement beyond what the numbers actually show. It
reuses the same L2/relative-L2 error metrics
(:mod:`femtoolkit.verification.metrics`) *verification* uses --
validation and verification ask different questions, but "how wrong is
this number" is answered the same mathematical way either time.
"""

from __future__ import annotations

import numpy as np

from femtoolkit.exceptions import ValidationError as ToolkitValidationError
from femtoolkit.validation.datasets import ReferenceDataset
from femtoolkit.validation.results import ValidationResult
from femtoolkit.verification.metrics import l2_error, relative_l2_error
from femtoolkit.verification.status import VerificationStatus
from femtoolkit.verification.tolerance import Tolerance

_DEFAULT_TOLERANCE = Tolerance(absolute=1e-6, relative=0.05)
"""Looser than the verification benchmarks' default tolerance
deliberately: a reference dataset (especially an experimental one) has
its own measurement/discretization uncertainty, so demanding
floating-point agreement from a validation comparison would be a
category error -- a few percent relative agreement is a realistic
default for physical validation; tighten or loosen it per dataset."""


def compare_to_reference_dataset(
    dataset: ReferenceDataset,
    simulation_values: np.ndarray,
    tolerance: Tolerance = _DEFAULT_TOLERANCE,
) -> ValidationResult:
    """Compare simulation values against a reference dataset's values.

    Args:
        dataset: The reference dataset to compare against.
        simulation_values: The FEA values to validate, in the same
            order and units as ``dataset.values`` (e.g. one value per
            ``dataset.independent_variable`` entry -- aligning them is
            the caller's responsibility; this function performs no
            interpolation or resampling).
        tolerance: The combined absolute/relative tolerance the L2
            difference must be within.

    Returns:
        A :class:`~femtoolkit.validation.results.ValidationResult`.

    Raises:
        ValidationError: If ``simulation_values`` does not have the
            same length as ``dataset.values``.
    """
    simulation_array = np.asarray(simulation_values, dtype=float)
    if simulation_array.shape != dataset.values.shape:
        raise ToolkitValidationError(
            f"simulation_values shape {simulation_array.shape} does not match "
            f"dataset '{dataset.name}' values shape {dataset.values.shape}."
        )

    abs_err = l2_error(simulation_array, dataset.values)
    rel_err = relative_l2_error(simulation_array, dataset.values)
    reference_norm = float(np.linalg.norm(dataset.values))
    satisfied = abs_err <= tolerance.allowed_error(reference_norm)
    status = VerificationStatus.PASS if satisfied else VerificationStatus.FAIL

    message = (
        f"{dataset.quantity} vs. '{dataset.name}' ({dataset.source}): "
        f"absolute_error={abs_err:.6e} {dataset.units}, relative_error={rel_err:.6e} "
        f"-> {status.value.upper()}"
    )

    return ValidationResult(
        dataset_name=dataset.name,
        dataset_source=dataset.source,
        quantity=dataset.quantity,
        simulation_values=simulation_array,
        reference_values=dataset.values,
        absolute_error=abs_err,
        relative_error=rel_err,
        tolerance=tolerance,
        status=status,
        message=message,
        has_uncertainty=dataset.uncertainty is not None,
    )


__all__ = ["compare_to_reference_dataset"]
