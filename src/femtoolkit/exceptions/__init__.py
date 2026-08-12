"""Custom exception types shared across the Finite Element Toolkit domain model."""

from femtoolkit.exceptions.exceptions import (
    DegenerateElementError,
    DuplicateIDError,
    EntityNotFoundError,
    FiniteElementToolkitError,
    InsufficientConstraintsError,
    InvalidAnalysisError,
    InvalidElementError,
    SingularSystemError,
    ValidationError,
)

__all__ = [
    "DegenerateElementError",
    "DuplicateIDError",
    "EntityNotFoundError",
    "FiniteElementToolkitError",
    "InsufficientConstraintsError",
    "InvalidAnalysisError",
    "InvalidElementError",
    "SingularSystemError",
    "ValidationError",
]
