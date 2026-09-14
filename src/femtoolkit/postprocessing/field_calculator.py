"""Derived post-processing quantities: equivalent measures, magnitudes, and deformed geometry.

This module computes only what is **not already available** from an
existing solver/result/element method, and everywhere an existing
formula already exists, it calls that formula directly rather than
re-deriving it (see :mod:`femtoolkit.continuum.stress`/
:mod:`femtoolkit.continuum.tensor`, already the single source of truth
for von Mises stress and principal stresses since Version 6/15):

* **Equivalent (von Mises) stress** dispatches to the existing
  :func:`~femtoolkit.continuum.stress.von_mises_plane_stress` (a 3-
  component 2D Voigt stress) or :func:`~femtoolkit.continuum.stress.von_mises_3d`
  (a 6-component 3D Voigt stress) by inspecting the stress vector's
  length -- no new stress math.
* **Equivalent (von Mises) strain** is genuinely new (no prior version
  needed it): the standard definition,

  .. code-block:: text

      epsilon_eq = sqrt(2/3 * e:e)

  where ``e`` is the deviatoric strain tensor, is implemented once in
  :func:`~femtoolkit.continuum.tensor.equivalent_strain_from_tensor`
  (added alongside its stress sibling,
  :func:`~femtoolkit.continuum.tensor.von_mises_stress_from_tensor`, in
  the same module) and simply called here after converting the
  engineering-shear Voigt strain to tensor form via the existing
  :func:`~femtoolkit.continuum.tensor.voigt_strain_to_tensor`. Scoped to
  3D (6-component) strain only, matching this project's existing
  precedent of scoping tensor-invariant quantities (e.g.
  :meth:`~femtoolkit.results.analysis_result.AnalysisResult.element_hydrostatic_stress`)
  to the 3D solid elements that have a full stress/strain tensor to
  begin with.
* **Vector magnitude** (for heat flux, temperature gradient, or
  displacement) is a plain Euclidean norm -- no formula to derive.
* **Deformed geometry** (Version 22's brief, section 11) is
  ``x_deformed = x_original + s * u``: a node's original coordinates
  plus its displacement, scaled by a *visualization* factor ``s`` that
  exists purely to make small physical displacements visible on a
  plot. ``s`` never appears in, and never alters, any physically
  computed result -- it is applied only to the copy of the coordinates
  handed to a plot, never fed back into any calculation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from femtoolkit.continuum.stress import von_mises_3d, von_mises_plane_stress
from femtoolkit.continuum.tensor import equivalent_strain_from_tensor, voigt_strain_to_tensor
from femtoolkit.exceptions import ValidationError
from femtoolkit.postprocessing.result_model import MeshTopology, ResultStep, SimulationResult


def equivalent_stress(stress_voigt: np.ndarray) -> float:
    """Return the von Mises equivalent stress from a Voigt stress vector.

    Dispatches by vector length: 3 components (``[sigma_x, sigma_y,
    tau_xy]``, a 2D continuum element) uses
    :func:`~femtoolkit.continuum.stress.von_mises_plane_stress`; 6
    components (``[sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz,
    tau_xz]``, a 3D solid element) uses
    :func:`~femtoolkit.continuum.stress.von_mises_3d`.

    Args:
        stress_voigt: A length-3 or length-6 Voigt stress vector, in pascals.

    Returns:
        The von Mises equivalent stress, in pascals.

    Raises:
        ValidationError: If ``stress_voigt`` has neither 3 nor 6 components.
    """
    values = np.asarray(stress_voigt, dtype=float)
    if values.size == 3:
        return von_mises_plane_stress(*values)
    if values.size == 6:
        return von_mises_3d(*values)
    raise ValidationError(
        f"equivalent_stress requires a length-3 or length-6 Voigt stress vector, "
        f"got length {values.size}."
    )


def equivalent_strain(strain_voigt: np.ndarray) -> float:
    """Return the von Mises equivalent strain from a 3D Voigt strain vector.

    See the module docstring for the formula and why this is scoped to
    3D (6-component, ``[epsilon_xx, epsilon_yy, epsilon_zz, gamma_xy,
    gamma_yz, gamma_xz]``) strain only.

    Args:
        strain_voigt: A length-6 Voigt strain vector (engineering shear).

    Returns:
        The von Mises equivalent strain (dimensionless).

    Raises:
        ValidationError: If ``strain_voigt`` does not have 6 components.
    """
    values = np.asarray(strain_voigt, dtype=float)
    if values.size != 6:
        raise ValidationError(
            f"equivalent_strain requires a length-6 Voigt strain vector, got length {values.size}."
        )
    return equivalent_strain_from_tensor(voigt_strain_to_tensor(values))


def vector_magnitude(vector: np.ndarray) -> float:
    """Return the Euclidean norm of a vector quantity (heat flux, gradient, displacement, ...)."""
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def deformed_coordinates(
    topology: MeshTopology, displacement_field: dict[int, np.ndarray], scale: float = 1.0
) -> dict[int, tuple[float, float, float]]:
    """Return every node's deformed position, ``x_deformed = x_original + scale * u``.

    Args:
        topology: The mesh topology providing original node coordinates.
        displacement_field: Maps each node ID to its displacement vector
            (length 1-3; missing trailing components are treated as zero).
        scale: The visualization scale factor ``s``. Defaults to ``1.0``
            (the true, unscaled displacement). Purely cosmetic: it never
            changes any physically computed result, only how far a
            plotted node is moved from its original position.

    Returns:
        Maps each node ID present in ``displacement_field`` to its
        deformed ``(x, y, z)`` position, in meters.
    """
    deformed: dict[int, tuple[float, float, float]] = {}
    for node_id, displacement in displacement_field.items():
        original = np.array(topology.node_coordinates[node_id])
        u = np.zeros(3)
        u[: len(displacement)] = displacement
        deformed[node_id] = tuple(original + scale * u)
    return deformed


_MAGNITUDE_FIELDS: dict[str, str] = {
    "temperature_gradient": "temperature_gradient_magnitude",
    "heat_flux": "heat_flux_magnitude",
    "displacement": "displacement_magnitude",
}


def with_derived_fields(result: SimulationResult) -> SimulationResult:
    """Return a copy of ``result`` enriched with derived scalar fields, computed once.

    Adds, to every step where the underlying field is present:

    * ``*_magnitude`` for every vector field named in
      :data:`_MAGNITUDE_FIELDS` (``temperature_gradient``, ``heat_flux``,
      ``displacement``) -- the Euclidean norm via :func:`vector_magnitude`.
    * ``von_mises_stress``, from the ``stress`` element field, if every
      stress vector at that step shares a consistent length (1, 3, or 6).
    * ``equivalent_strain``, from the ``strain`` element field, if every
      strain vector at that step has exactly 6 components (see the
      module docstring for why this is 3D-only).

    This is the toolkit's **Field Calculator** step in the Version 22
    post-processing pipeline: it runs once, after a
    :mod:`femtoolkit.postprocessing.adapters` call and before
    visualization/export/summarization, so derived quantities are
    computed exactly once rather than recomputed by every consumer.

    Args:
        result: The (raw) simulation result to enrich.

    Returns:
        A new :class:`~femtoolkit.postprocessing.result_model.SimulationResult`
        with the same steps, plus the derived fields described above.
    """
    enriched_steps = []
    for step in result.steps:
        nodal_fields = dict(step.nodal_fields)
        element_fields = dict(step.element_fields)

        for source_name, target_name in _MAGNITUDE_FIELDS.items():
            if source_name in nodal_fields:
                nodal_fields[target_name] = {
                    entity_id: vector_magnitude(value)
                    for entity_id, value in nodal_fields[source_name].items()
                }
            if source_name in element_fields:
                element_fields[target_name] = {
                    entity_id: vector_magnitude(value)
                    for entity_id, value in element_fields[source_name].items()
                }

        if "stress" in element_fields:
            sizes = {np.asarray(value).size for value in element_fields["stress"].values()}
            if len(sizes) == 1 and sizes.pop() in (1, 3, 6):
                element_fields["von_mises_stress"] = {
                    entity_id: _safe_equivalent_stress(value)
                    for entity_id, value in element_fields["stress"].items()
                }

        if "strain" in element_fields:
            sizes = {np.asarray(value).size for value in element_fields["strain"].values()}
            if sizes == {6}:
                element_fields["equivalent_strain"] = {
                    entity_id: equivalent_strain(value)
                    for entity_id, value in element_fields["strain"].items()
                }

        enriched_steps.append(
            ResultStep(
                step.index, step.time, nodal_fields=nodal_fields, element_fields=element_fields
            )
        )

    return SimulationResult(
        result.topology, tuple(enriched_steps), field_units=dict(result.field_units)
    )


@dataclass(frozen=True)
class EngineeringSummary:
    """A lightweight numerical summary of a
    :class:`~femtoolkit.postprocessing.result_model.SimulationResult`.

    Every field is ``None`` when the corresponding quantity is not
    present in the result (e.g. ``maximum_von_mises_stress`` is ``None``
    for a purely thermal result).

    Attributes:
        minimum_temperature: Minimum nodal temperature over every step, in kelvin.
        maximum_temperature: Maximum nodal temperature over every step, in kelvin.
        maximum_heat_flux: Maximum heat-flux magnitude over every step, in W/m^2.
        maximum_displacement: Maximum displacement magnitude over every step, in meters.
        maximum_von_mises_stress: Maximum von Mises equivalent stress over every step, in pascals.
        maximum_equivalent_strain: Maximum von Mises equivalent strain over every step.
        final_time: The last step's :attr:`~femtoolkit.postprocessing.result_model.ResultStep.time`.
        num_steps: The number of steps in the result.
    """

    minimum_temperature: float | None
    maximum_temperature: float | None
    maximum_heat_flux: float | None
    maximum_displacement: float | None
    maximum_von_mises_stress: float | None
    maximum_equivalent_strain: float | None
    final_time: float
    num_steps: int


def _max_nodal_scalar(result: SimulationResult, field_name: str) -> float | None:
    if field_name not in result.final_step.nodal_fields:
        return None
    return max(
        float(value)
        for step in result.steps
        for value in step.nodal_field(field_name).values()
    )


def _min_nodal_scalar(result: SimulationResult, field_name: str) -> float | None:
    if field_name not in result.final_step.nodal_fields:
        return None
    return min(
        float(value)
        for step in result.steps
        for value in step.nodal_field(field_name).values()
    )


def _max_nodal_magnitude(result: SimulationResult, field_name: str) -> float | None:
    if field_name not in result.final_step.nodal_fields:
        return None
    return max(
        vector_magnitude(value)
        for step in result.steps
        for value in step.nodal_field(field_name).values()
    )


def _max_element_magnitude(result: SimulationResult, field_name: str) -> float | None:
    if field_name not in result.final_step.element_fields:
        return None
    return max(
        vector_magnitude(value)
        for step in result.steps
        for value in step.element_field(field_name).values()
    )


def _max_element_equivalent(
    result: SimulationResult, field_name: str, formula: Callable[[np.ndarray], float]
) -> float | None:
    if field_name not in result.final_step.element_fields:
        return None
    return max(
        formula(value)
        for step in result.steps
        for value in step.element_field(field_name).values()
    )


def _safe_equivalent_strain(strain_voigt: np.ndarray) -> float:
    """Return the equivalent strain, or 0.0 for a non-6-component (e.g. 2D) strain vector.

    Used only by :func:`summarize`, which must summarize a result
    without knowing in advance whether its strain field is 2D or 3D --
    see the module docstring for why equivalent strain itself is scoped
    to 3D.
    """
    values = np.asarray(strain_voigt, dtype=float)
    if values.size != 6:
        return 0.0
    return equivalent_strain(values)


def _safe_equivalent_stress(stress_voigt: np.ndarray) -> float:
    """Return an equivalent stress measure for any stress shape a result might carry.

    A length-1 (scalar) stress -- a 1D structural member (bar/truss) --
    has no shear to combine, so its own magnitude already *is* the
    equivalent stress; length-3/6 dispatch to :func:`equivalent_stress`.
    Used only by :func:`summarize`, which must summarize a result
    without knowing its element type in advance.
    """
    values = np.asarray(stress_voigt, dtype=float)
    if values.size == 1:
        return abs(float(values))
    return equivalent_stress(values)


def summarize(result: SimulationResult) -> EngineeringSummary:
    """Compute a lightweight engineering summary of every recognized field in ``result``.

    Args:
        result: The simulation result to summarize.

    Returns:
        An :class:`EngineeringSummary`.
    """
    return EngineeringSummary(
        minimum_temperature=_min_nodal_scalar(result, "temperature"),
        maximum_temperature=_max_nodal_scalar(result, "temperature"),
        maximum_heat_flux=_max_element_magnitude(result, "heat_flux"),
        maximum_displacement=_max_nodal_magnitude(result, "displacement"),
        maximum_von_mises_stress=_max_element_equivalent(
            result, "stress", _safe_equivalent_stress
        ),
        maximum_equivalent_strain=_max_element_equivalent(
            result, "strain", _safe_equivalent_strain
        ),
        final_time=float(result.final_step.time),
        num_steps=result.num_steps,
    )
