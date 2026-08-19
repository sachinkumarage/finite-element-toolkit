"""A named collection of solved analysis results, one per load case/combination.

:class:`ResultSet` is what :meth:`~femtoolkit.analysis.load_manager.LoadManager.solve_all`
returns: every registered load case and load combination, already solved,
addressable by name:

.. code-block:: text

    results.for_load_case("Dead Load")
    results.for_load_case("Wind Load")
    results.for_combination("Ultimate")

Each value is an ordinary :class:`~femtoolkit.results.analysis_result.AnalysisResult`
-- displacements, reactions, and per-element strain/stress/stress
recovery all work exactly as they do for a single, individually solved
analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

from femtoolkit.exceptions import EntityNotFoundError
from femtoolkit.results.analysis_result import AnalysisResult


@dataclass(frozen=True)
class ResultSet:
    """Read-only collection of solved results, keyed by load case/combination name.

    Instances are produced by
    :meth:`~femtoolkit.analysis.load_manager.LoadManager.solve_all` and
    should not be constructed directly by application code.

    Attributes:
        load_case_results: Maps each solved load case's name to its
            :class:`~femtoolkit.results.analysis_result.AnalysisResult`.
        combination_results: Maps each solved load combination's name to
            its :class:`~femtoolkit.results.analysis_result.AnalysisResult`.
    """

    load_case_results: dict[str, AnalysisResult]
    combination_results: dict[str, AnalysisResult]

    def for_load_case(self, name: str) -> AnalysisResult:
        """Return the solved result for a named load case.

        Args:
            name: The load case's name, as passed to
                ``LoadCase(name=...)``.

        Returns:
            The load case's :class:`~femtoolkit.results.analysis_result.AnalysisResult`.

        Raises:
            EntityNotFoundError: If no load case with that name was
                solved into this result set.
        """
        try:
            return self.load_case_results[name]
        except KeyError as error:
            raise EntityNotFoundError(
                f"No load case named {name!r} in this result set."
            ) from error

    def for_combination(self, name: str) -> AnalysisResult:
        """Return the solved result for a named load combination.

        Args:
            name: The combination's name, as passed to
                ``LoadCombination(name=...)``.

        Returns:
            The combination's :class:`~femtoolkit.results.analysis_result.AnalysisResult`.

        Raises:
            EntityNotFoundError: If no combination with that name was
                solved into this result set.
        """
        try:
            return self.combination_results[name]
        except KeyError as error:
            raise EntityNotFoundError(
                f"No load combination named {name!r} in this result set."
            ) from error
