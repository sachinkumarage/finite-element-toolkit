"""Centralized load-case/load-combination management for one mesh.

:class:`LoadManager` is the top of the Version 10 loading workflow: it
holds a single mesh, collects named load cases and load combinations
registered against it, and solves all of them in one call:

.. code-block:: text

    manager = LoadManager(mesh)
    manager.add_load_case(dead_load)
    manager.add_load_case(live_load)
    manager.add_combination(ultimate)
    results = manager.solve_all()

    results.for_load_case("Dead Load")
    results.for_combination("Ultimate")

Each load case/combination is solved independently (its own
:class:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis`, its own
:class:`~femtoolkit.results.analysis_result.AnalysisResult`) -- this
module performs no analysis of its own, only bookkeeping and dispatch.
"""

from __future__ import annotations

from femtoolkit.analysis.load_case import LoadCase
from femtoolkit.analysis.load_combination import LoadCombination
from femtoolkit.exceptions import ValidationError
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.results.result_set import ResultSet


class LoadManager:
    """Registers load cases and load combinations against one mesh, and solves them.

    Example:
        >>> manager = LoadManager(mesh)
        >>> manager.add_load_case(dead_load)
        >>> manager.add_load_case(live_load)
        >>> manager.add_combination(ultimate)
        >>> results = manager.solve_all()
        >>> results.for_combination("Ultimate")
    """

    def __init__(self, mesh: Mesh) -> None:
        """Create an empty load manager for the given mesh.

        Args:
            mesh: The mesh every registered load case/combination will be
                solved against.
        """
        self.mesh = mesh
        self._load_cases: dict[str, LoadCase] = {}
        self._combinations: dict[str, LoadCombination] = {}

    def add_load_case(self, load_case: LoadCase) -> None:
        """Register a load case under its own name.

        Args:
            load_case: The load case to register. Its own ``mesh`` (if
                bound) does not need to match this manager's mesh -- it
                is solved against this manager's mesh regardless (see
                :meth:`~femtoolkit.analysis.load_case.LoadCase.solve`).

        Raises:
            ValidationError: If a load case with the same name has
                already been registered.
        """
        if load_case.name in self._load_cases:
            raise ValidationError(
                f"A load case named {load_case.name!r} has already been registered."
            )
        self._load_cases[load_case.name] = load_case

    def add_combination(self, combination: LoadCombination) -> None:
        """Register a load combination under its own name.

        Args:
            combination: The load combination to register.

        Raises:
            ValidationError: If a combination with the same name has
                already been registered.
        """
        if combination.name in self._combinations:
            raise ValidationError(
                f"A load combination named {combination.name!r} has already been registered."
            )
        self._combinations[combination.name] = combination

    def solve_all(self) -> ResultSet:
        """Solve every registered load case and load combination against this mesh.

        Returns:
            A :class:`~femtoolkit.results.result_set.ResultSet` holding
            every solved result, addressable by name.
        """
        load_case_results = {
            name: load_case.solve(self.mesh) for name, load_case in self._load_cases.items()
        }
        combination_results = {
            name: combination.solve(self.mesh)
            for name, combination in self._combinations.items()
        }
        return ResultSet(
            load_case_results=load_case_results, combination_results=combination_results
        )
