"""User-facing error message formatting (Version 24, spec section 17).

The GUI must "distinguish between user/input errors, model errors,
solver errors, and visualization errors" and "not expose unnecessary
Python tracebacks to normal users." :func:`describe_error` is the one
place that maps a caught exception to a short, professional message;
every ``except FiniteElementToolkitError`` in :mod:`femtoolkit.gui`
routes through it rather than formatting ``str(exc)`` ad hoc. No
traceback is ever included -- normal (non-caught) exception propagation
outside the GUI remains available for development/debugging, unchanged.
"""

from __future__ import annotations

from femtoolkit.exceptions import (
    EntityNotFoundError,
    FiniteElementToolkitError,
    InsufficientConstraintsError,
    NonlinearConvergenceError,
    SingularSystemError,
    ValidationError,
)


def describe_error(exc: Exception) -> str:
    """Turn a caught exception into a short, professional, user-facing message.

    Args:
        exc: The caught exception.

    Returns:
        A one-line message, prefixed by its category (input, model, or
        solver), suitable for direct display in the GUI.
    """
    if isinstance(exc, ValidationError):
        return f"Invalid input: {exc}"
    if isinstance(exc, EntityNotFoundError):
        return f"Model error: {exc}"
    if isinstance(
        exc, (SingularSystemError, InsufficientConstraintsError, NonlinearConvergenceError)
    ):
        return f"Solver error: {exc}"
    if isinstance(exc, FiniteElementToolkitError):
        return f"Error: {exc}"
    return f"Unexpected error: {exc}"


def describe_visualization_error(exc: Exception) -> str:
    """Turn a caught visualization-layer exception into a user-facing message.

    Args:
        exc: The caught exception (typically an ``ImportError`` for a
            missing PyVista installation, or a
            :class:`~femtoolkit.exceptions.ValidationError` for an
            unavailable result field).

    Returns:
        A one-line, user-facing message.
    """
    if isinstance(exc, ImportError):
        return (
            "3D visualization is unavailable: PyVista is not installed. "
            'Install it with: pip install "femtoolkit[viz3d]"'
        )
    if isinstance(exc, ValidationError):
        return f"Visualization error: the selected result field is unavailable. {exc}"
    return f"Visualization error: {exc}"
