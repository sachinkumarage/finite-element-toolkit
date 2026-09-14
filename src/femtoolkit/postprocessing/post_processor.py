"""Query operations over a solved :class:`~femtoolkit.postprocessing.result_model.SimulationResult`.

This is the "ask a question about the result" layer (statistics, value
lookups, time/load-step histories, engineering summaries) -- distinct
from :mod:`femtoolkit.postprocessing.field_calculator`, which *computes*
values that were not already in the result, and from
:mod:`femtoolkit.postprocessing.visualization`, which turns query
results into plots. :class:`PostProcessor` performs no solving and adds
no new field data; every method here is a read-only view over what a
:class:`~femtoolkit.postprocessing.result_model.SimulationResult`
already contains.

**Scalarizing a vector field.** ``minimum``/``maximum``/``mean`` need a
single number even when the underlying field is vector-valued (a
displacement, a heat flux). Given an explicit ``component``, that one
component is used directly (e.g. ``component=0`` for ``ux``); with no
``component``, the field's **magnitude** (Euclidean norm) is used --
the same convention
:func:`~femtoolkit.postprocessing.field_calculator.summarize` already
uses for its own "maximum displacement"/"maximum heat flux" entries, so
a caller asking "what is the maximum displacement" without specifying a
direction gets the physically meaningful resultant, not an arbitrary
component.
"""

from __future__ import annotations

import numpy as np

from femtoolkit.exceptions import ValidationError
from femtoolkit.postprocessing.field_calculator import (
    EngineeringSummary,
    summarize,
    vector_magnitude,
)
from femtoolkit.postprocessing.result_model import FieldValue, SimulationResult


def _scalarize(value: FieldValue, component: int | None) -> float:
    if component is not None:
        return float(np.asarray(value)[component])
    if np.isscalar(value) or np.asarray(value).ndim == 0:
        return float(value)
    return vector_magnitude(value)


class PostProcessor:
    """Query operations (statistics, lookups, histories) over a solved simulation result.

    Attributes:
        result: The wrapped :class:`~femtoolkit.postprocessing.result_model.SimulationResult`.

    Example:
        >>> processor = PostProcessor(result)
        >>> processor.maximum("temperature")
        410.0
        >>> processor.history("temperature", node_id=3)
        array([293.15, 350.2, 410.0])
    """

    def __init__(self, result: SimulationResult) -> None:
        """Wrap a solved simulation result for querying.

        Args:
            result: The result to query.
        """
        self.result = result

    def _field_values(
        self, field_name: str, step: int, kind: str
    ) -> dict[int, FieldValue]:
        result_step = self.result.step(step)
        if kind == "nodal":
            return result_step.nodal_field(field_name)
        if kind == "element":
            return result_step.element_field(field_name)
        raise ValidationError(f'kind must be "nodal" or "element", got {kind!r}.')

    def minimum(
        self, field_name: str, step: int = -1, component: int | None = None, kind: str = "nodal"
    ) -> float:
        """Return the minimum value of a field over every entity at one step.

        Args:
            field_name: The field to query.
            step: Which step to query (default: the last).
            component: Which vector component to use, or ``None`` for
                the magnitude of a vector field (see the module docstring).
            kind: ``"nodal"`` (default) or ``"element"``.

        Returns:
            The minimum scalar value.
        """
        values = self._field_values(field_name, step, kind)
        return min(_scalarize(value, component) for value in values.values())

    def maximum(
        self, field_name: str, step: int = -1, component: int | None = None, kind: str = "nodal"
    ) -> float:
        """Return the maximum value of a field over every entity at one step.

        See :meth:`minimum` for the argument semantics.
        """
        values = self._field_values(field_name, step, kind)
        return max(_scalarize(value, component) for value in values.values())

    def mean(
        self, field_name: str, step: int = -1, component: int | None = None, kind: str = "nodal"
    ) -> float:
        """Return the mean value of a field over every entity at one step. See :meth:`minimum`."""
        values = self._field_values(field_name, step, kind)
        scalars = [_scalarize(value, component) for value in values.values()]
        return float(np.mean(scalars))

    def value_range(
        self, field_name: str, step: int = -1, component: int | None = None, kind: str = "nodal"
    ) -> tuple[float, float]:
        """Return ``(minimum, maximum)`` of a field over every entity at one step."""
        return (
            self.minimum(field_name, step, component, kind),
            self.maximum(field_name, step, component, kind),
        )

    def nodal_values(self, field_name: str, step: int = -1) -> dict[int, FieldValue]:
        """Return the ``{node_id: value}`` mapping for a nodal field at one step."""
        return self.result.step(step).nodal_field(field_name)

    def element_values(self, field_name: str, step: int = -1) -> dict[int, FieldValue]:
        """Return the ``{element_id: value}`` mapping for an element field at one step."""
        return self.result.step(step).element_field(field_name)

    def values_at_nodes(
        self, field_name: str, node_ids: list[int], step: int = -1
    ) -> dict[int, FieldValue]:
        """Return one field's value at a chosen subset of nodes, at one step.

        Args:
            field_name: The nodal field to query.
            node_ids: The nodes to extract.
            step: Which step to query (default: the last).

        Returns:
            Maps each requested node ID to its value.
        """
        result_step = self.result.step(step)
        return {node_id: result_step.nodal_value(field_name, node_id) for node_id in node_ids}

    def history(
        self,
        field_name: str,
        node_id: int | None = None,
        element_id: int | None = None,
        component: int | None = None,
    ) -> np.ndarray:
        """Return one entity's value for one field across every step (time or load history).

        Args:
            field_name: The field to extract.
            node_id: The node to extract a nodal field's history for.
                Exactly one of ``node_id``/``element_id`` must be given.
            element_id: The element to extract an element field's
                history for.
            component: Which vector component to extract, or ``None``
                for a scalar field.

        Returns:
            A NumPy array of shape ``(num_steps,)``.

        Raises:
            ValidationError: If both or neither of ``node_id``/
                ``element_id`` are given.
        """
        if (node_id is None) == (element_id is None):
            raise ValidationError("history requires exactly one of node_id or element_id.")
        if node_id is not None:
            return self.result.nodal_history(field_name, node_id, component)
        return self.result.element_history(field_name, element_id, component)

    def summary(self) -> EngineeringSummary:
        """Return a lightweight engineering summary of the wrapped result.

        See :func:`~femtoolkit.postprocessing.field_calculator.summarize`.
        """
        return summarize(self.result)
