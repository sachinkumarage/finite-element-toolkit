"""Queries a solved simulation result for the results dashboard (Version 24).

A thin wrapper around Version 22's
:class:`~femtoolkit.postprocessing.post_processor.PostProcessor` --
section 13 of this version's brief is explicit that the results
dashboard must "only display quantities that actually exist in the
result object," so this service exposes exactly which fields a given
result carries (:meth:`available_fields`) rather than assuming a fixed
mechanical or thermal field set.
"""

from __future__ import annotations

from dataclasses import dataclass

from femtoolkit.postprocessing import EngineeringSummary, PostProcessor, SimulationResult, summarize


@dataclass(frozen=True)
class FieldRange:
    """The minimum and maximum of one field over a result's final step.

    Attributes:
        field_name: The field this range describes.
        minimum: The minimum value.
        maximum: The maximum value.
    """

    field_name: str
    minimum: float
    maximum: float


class ResultsService:
    """Queries a solved :class:`~femtoolkit.postprocessing.result_model.SimulationResult`."""

    def summary(self, simulation: SimulationResult) -> EngineeringSummary:
        """Return the engineering summary (min/max temperature, stress, displacement, ...)."""
        return summarize(simulation)

    def available_nodal_fields(self, simulation: SimulationResult) -> list[str]:
        """Return every nodal field name present on the result's final step."""
        return sorted(simulation.final_step.nodal_fields)

    def available_element_fields(self, simulation: SimulationResult) -> list[str]:
        """Return every element field name present on the result's final step."""
        return sorted(simulation.final_step.element_fields)

    def field_range(
        self, simulation: SimulationResult, field_name: str, kind: str = "nodal"
    ) -> FieldRange:
        """Return one field's minimum and maximum value at the final step.

        Args:
            simulation: The result to query.
            field_name: The field to summarize.
            kind: ``"nodal"`` or ``"element"``.

        Returns:
            A :class:`FieldRange`.

        Raises:
            ValueError: If ``kind`` is not ``"nodal"``/``"element"``.
        """
        processor = PostProcessor(simulation)
        if kind not in ("nodal", "element"):
            raise ValueError(f"kind must be 'nodal' or 'element', got {kind!r}.")
        minimum = processor.minimum(field_name, kind=kind)
        maximum = processor.maximum(field_name, kind=kind)
        return FieldRange(field_name=field_name, minimum=minimum, maximum=maximum)
