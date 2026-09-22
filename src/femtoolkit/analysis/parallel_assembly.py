"""Parallel element-level contribution computation (Version 27).

Global assembly (:mod:`femtoolkit.analysis.assembly`,
:mod:`femtoolkit.analysis.sparse_assembly`) has always taken a plain
``Sequence`` of already-computed element contributions and scattered them
into a global matrix -- a fast, inherently sequential accumulation step
(spec section 10/11: never let several workers write into the same
global matrix at once). What *can* run independently, one element at a
time, is computing each contribution in the first place: an element's
``stiffness_matrix``/conductivity matrix depends only on that element's
own geometry, material, and (for the thermal case) the evaluation
temperature -- never on any other element or on the global DOF numbering.

.. code-block:: text

    Element 1 -> stiffness_matrix ─┐
    Element 2 -> stiffness_matrix ─┼─>  [contributions]  -> assemble_global_stiffness(...)
    Element 3 -> stiffness_matrix ─┘        (sequential)      (sequential, already fast)
         (independent, parallelizable)

This module supplies exactly that "compute every element's contribution"
step, through an :class:`~femtoolkit.execution.executor.ElementExecutor`
-- serial (the default, byte-for-byte the same work
:class:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis` and
:class:`~femtoolkit.thermal.thermal_analysis.SteadyStateThermalAnalysis`
already did as a list comprehension through Version 26) or parallel. The
resulting contributions feed the exact same, unmodified
:func:`~femtoolkit.analysis.assembly.assemble_global_stiffness`/
:func:`~femtoolkit.analysis.sparse_assembly.assemble_global_stiffness_sparse`
either way -- this module never constructs a global matrix itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from femtoolkit.analysis.assembly import ElementStiffnessContribution
from femtoolkit.analysis.element import AssemblableElement

if TYPE_CHECKING:
    from femtoolkit.execution.executor import ElementExecutor
    from femtoolkit.thermal.thermal_elements import ThermalCapableElement
    from femtoolkit.thermal.thermal_material import ThermalMaterial


def _stiffness_contribution(element: AssemblableElement) -> ElementStiffnessContribution:
    """Compute one element's stiffness contribution.

    A module-level function (rather than a lambda or a
    :class:`StaticLinearAnalysis` method) so it can be pickled and sent
    to a worker process by a
    :class:`~femtoolkit.execution.parallel.ParallelExecutor` using the
    ``"process"`` backend.
    """
    return ElementStiffnessContribution(element.dof_keys(), element.stiffness_matrix)


def compute_stiffness_contributions(
    elements: Sequence[AssemblableElement], executor: ElementExecutor
) -> list[ElementStiffnessContribution]:
    """Compute every element's stiffness contribution, via ``executor``.

    Args:
        elements: The mesh's elements, each satisfying
            :class:`~femtoolkit.analysis.element.AssemblableElement`.
        executor: The
            :class:`~femtoolkit.execution.executor.ElementExecutor`
            strategy to use (serial or parallel).

    Returns:
        One :class:`~femtoolkit.analysis.assembly.ElementStiffnessContribution`
        per element, in the same order as ``elements`` -- ready for
        :func:`~femtoolkit.analysis.assembly.assemble_global_stiffness`
        or :func:`~femtoolkit.analysis.sparse_assembly.assemble_global_stiffness_sparse`.

    Raises:
        TaskSerializationError: If a parallel ``executor`` using the
            ``"process"`` backend cannot pickle an element.
        WorkerExecutionError: If computing an element's stiffness matrix
            raises inside a worker.
    """
    return executor.map(_stiffness_contribution, elements)


def _conductivity_task(
    task: tuple[ThermalCapableElement, ThermalMaterial, float],
) -> ElementStiffnessContribution:
    """Compute one element's conductivity contribution from a packed task tuple.

    A single-argument, module-level wrapper around
    :func:`~femtoolkit.thermal.thermal_elements.conductivity_contribution`
    (which itself takes three arguments) -- an
    :class:`~femtoolkit.execution.executor.ElementExecutor` only maps
    one-argument callables over one sequence of items.
    """
    from femtoolkit.thermal.thermal_elements import conductivity_contribution

    element, material, temperature = task
    return conductivity_contribution(element, material, temperature)


def compute_conductivity_contributions(
    elements: Sequence[ThermalCapableElement],
    materials: dict[int, ThermalMaterial],
    evaluation_temperature: float,
    executor: ElementExecutor,
) -> list[ElementStiffnessContribution]:
    """Compute every element's conductivity contribution, via ``executor``.

    Args:
        elements: The mesh's thermally-capable elements.
        materials: Maps each element's ``id`` to its
            :class:`~femtoolkit.thermal.thermal_material.ThermalMaterial`
            (every element in ``elements`` must have an entry -- the same
            precondition
            :class:`~femtoolkit.thermal.thermal_analysis.SteadyStateThermalAnalysis`
            already validates before calling this).
        evaluation_temperature: Temperature to evaluate
            temperature-dependent conductivity at, in kelvin.
        executor: The
            :class:`~femtoolkit.execution.executor.ElementExecutor`
            strategy to use (serial or parallel).

    Returns:
        One :class:`~femtoolkit.analysis.assembly.ElementStiffnessContribution`
        per element, in the same order as ``elements``.

    Raises:
        TaskSerializationError: If a parallel ``executor`` using the
            ``"process"`` backend cannot pickle an element or material.
        WorkerExecutionError: If computing an element's conductivity
            matrix raises inside a worker.
    """
    tasks = [(element, materials[element.id], evaluation_temperature) for element in elements]
    return executor.map(_conductivity_task, tasks)


__all__ = [
    "compute_conductivity_contributions",
    "compute_stiffness_contributions",
]
