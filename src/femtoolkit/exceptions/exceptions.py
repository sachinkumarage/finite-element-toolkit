"""Custom exception types for the Finite Element Toolkit.

These exceptions give callers a way to distinguish domain-level errors
(invalid engineering data, broken model references) from generic Python
exceptions, without introducing a separate exception type for every
possible failure.
"""

from __future__ import annotations


class FiniteElementToolkitError(Exception):
    """Base class for all errors raised by the Finite Element Toolkit.

    Catching this exception is a convenient way for calling code to
    handle any toolkit-specific failure without needing to know about
    every individual exception subclass.
    """


class ValidationError(FiniteElementToolkitError):
    """Raised when engineering or model data fails a validation rule.

    Examples include a negative material density, a non-finite node
    coordinate, or an element that references an empty node list.
    """


class DuplicateIDError(FiniteElementToolkitError):
    """Raised when an entity is added to a container under an ID that is
    already in use.

    For example, adding two nodes with the same ID to a :class:`Mesh`.
    """


class EntityNotFoundError(FiniteElementToolkitError):
    """Raised when a requested entity cannot be found in a container.

    For example, looking up a node ID that has not been added to the
    :class:`Mesh`.
    """
