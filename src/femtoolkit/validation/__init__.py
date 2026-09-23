"""FEA validation: "are we solving the correct physical problem?" (Version 29).

**Validation is not verification.**
:mod:`femtoolkit.verification` answers "are we solving the equations
correctly?" by comparing against *known analytical solutions* this
toolkit computes from closed-form formulas -- no external data is
needed. Validation answers a different question, and can only be
answered with *external, trusted reference or experimental evidence*:
this package provides the infrastructure to load such a dataset
(:class:`~femtoolkit.validation.datasets.ReferenceDataset`) and compare
simulation results against it
(:func:`~femtoolkit.validation.comparison.compare_to_reference_dataset`),
but it ships with **no bundled dataset** and performs **no
validation of its own** -- validation only happens when a caller
supplies real reference data. A simulation this toolkit has not been
given reference data for is *unvalidated*, not "validated by default";
this package never implies otherwise. See ``docs/validation.md`` for
the full guide.
"""

from __future__ import annotations

from femtoolkit.validation.comparison import compare_to_reference_dataset
from femtoolkit.validation.datasets import (
    ReferenceDataset,
    load_reference_dataset_csv,
    load_reference_dataset_json,
    reference_dataset_from_dict,
)
from femtoolkit.validation.results import ValidationResult

__all__ = [
    "ReferenceDataset",
    "ValidationResult",
    "compare_to_reference_dataset",
    "load_reference_dataset_csv",
    "load_reference_dataset_json",
    "reference_dataset_from_dict",
]
