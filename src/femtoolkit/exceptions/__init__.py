"""Custom exception types shared across the Finite Element Toolkit domain model."""

from femtoolkit.exceptions.exceptions import (
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
    "DuplicateIDError",
    "EntityNotFoundError",
    "FiniteElementToolkitError",
    "InsufficientConstraintsError",
    "InvalidAnalysisError",
    "InvalidElementError",
    "SingularSystemError",
    "ValidationError",
]
