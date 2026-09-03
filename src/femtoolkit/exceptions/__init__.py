"""Custom exception types shared across the Finite Element Toolkit domain model."""

from femtoolkit.exceptions.exceptions import (
    ConstitutiveUpdateError,
    DegenerateElementError,
    DuplicateIDError,
    DuplicateNodeCoordinatesError,
    EigenvalueComputationError,
    EntityNotFoundError,
    FiniteElementToolkitError,
    InsufficientConstraintsError,
    InvalidAnalysisError,
    InvalidDeformationGradientError,
    InvalidElementError,
    InvalidMaterialStateError,
    NonlinearConvergenceError,
    SingularSystemError,
    UnsupportedLoadingPathError,
    ValidationError,
)

__all__ = [
    "ConstitutiveUpdateError",
    "DegenerateElementError",
    "DuplicateIDError",
    "DuplicateNodeCoordinatesError",
    "EigenvalueComputationError",
    "EntityNotFoundError",
    "FiniteElementToolkitError",
    "InsufficientConstraintsError",
    "InvalidAnalysisError",
    "InvalidDeformationGradientError",
    "InvalidElementError",
    "InvalidMaterialStateError",
    "NonlinearConvergenceError",
    "SingularSystemError",
    "UnsupportedLoadingPathError",
    "ValidationError",
]
