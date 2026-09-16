"""A unified simulation-result data model, independent of which solver produced it (Version 22).

By Version 21, this toolkit has *five* distinct result classes --
:class:`~femtoolkit.results.analysis_result.AnalysisResult` (static
linear), :class:`~femtoolkit.results.nonlinear_result.NonlinearAnalysisResult`
(load-stepped), :class:`~femtoolkit.results.dynamic_result.DynamicResult`
(time history), :class:`~femtoolkit.thermal.thermal_result.SteadyStateThermalResult`,
and :class:`~femtoolkit.thermal.thermal_result.TransientThermalResult` --
each shaped around exactly what its own solver naturally produces (a
single displacement vector, a sequence of load steps, a time history of
displacement/velocity/acceleration, a single temperature vector, a time
history of temperature). That heterogeneity is the right choice *for
those classes* (see each module's own docstring for why it is not a
subclass of a shared base) -- but it is the wrong shape for
post-processing, which wants to ask the *same* question ("what is the
maximum value of field X at step Y") regardless of which solver produced
the data.

:class:`SimulationResult` is that common shape: a mesh's topology
(nodes, elements, connectivity, coordinates) plus an ordered sequence of
:class:`ResultStep`, each holding named nodal/element field values. It
is built *from* the existing result classes by the adapter functions in
:mod:`femtoolkit.postprocessing.adapters` -- this module performs no
solving and no constitutive/field calculation of its own; it is a pure,
passive container. A "step" means a load increment for a nonlinear
mechanical result, a time instant for a dynamic or transient thermal
result, or simply the (only) step of a static/steady-state result --
the same three-field model
(:attr:`ResultStep.index`/:attr:`ResultStep.time`/field dictionaries)
represents all of them uniformly.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from femtoolkit.exceptions import EntityNotFoundError, ValidationError

if TYPE_CHECKING:
    from femtoolkit.analysis.element import AssemblableElement
    from femtoolkit.mesh.mesh import Mesh

FieldValue = float | np.ndarray
"""One entity's value for one field: a scalar (e.g. temperature) or a
small vector (e.g. displacement, heat flux, a Voigt stress/strain)."""


@dataclass(frozen=True)
class MeshTopology:
    """The geometric/connectivity information a result needs to be visualized or exported.

    A deliberately minimal, solver-independent snapshot of a
    :class:`~femtoolkit.mesh.mesh.Mesh`: just enough to plot or export a
    result (coordinates, connectivity) without this package needing to
    depend on every concrete element class.

    Attributes:
        node_ids: Every node's ID, in a fixed order.
        node_coordinates: Maps each node ID to its ``(x, y, z)`` position,
            in meters.
        element_ids: Every element's ID, in a fixed order.
        element_connectivity: Maps each element ID to the tuple of node
            IDs it connects, in the element's own local node order.
        element_types: Maps each element ID to its element class name
            (e.g. ``"CSTElement2D"``), for labeling and dispatch.

    Raises:
        ValidationError: If ``node_ids``/``node_coordinates`` or
            ``element_ids``/``element_connectivity``/``element_types``
            are inconsistent with each other.
    """

    node_ids: tuple[int, ...]
    node_coordinates: dict[int, tuple[float, float, float]]
    element_ids: tuple[int, ...]
    element_connectivity: dict[int, tuple[int, ...]]
    element_types: dict[int, str]

    def __post_init__(self) -> None:
        """Validate internal consistency immediately after construction.

        Raises:
            ValidationError: If any node/element ID is missing from its
                corresponding lookup dictionary.
        """
        if set(self.node_ids) != set(self.node_coordinates):
            raise ValidationError(
                "MeshTopology node_ids and node_coordinates must reference the same nodes."
            )
        if set(self.element_ids) != set(self.element_connectivity) or set(
            self.element_ids
        ) != set(self.element_types):
            raise ValidationError(
                "MeshTopology element_ids must match element_connectivity and element_types."
            )

    @classmethod
    def from_mesh(cls, mesh: Mesh) -> MeshTopology:
        """Build a :class:`MeshTopology` snapshot from a live :class:`~femtoolkit.mesh.mesh.Mesh`.

        Args:
            mesh: The mesh to snapshot.

        Returns:
            A new :class:`MeshTopology`.
        """
        node_ids = tuple(node.id for node in mesh.nodes)
        node_coordinates = {node.id: (node.x, node.y, node.z) for node in mesh.nodes}
        element_ids = tuple(element.id for element in mesh.elements)
        element_connectivity = {
            element.id: tuple(node.id for node in element.nodes) for element in mesh.elements
        }
        element_types = {element.id: type(element).__name__ for element in mesh.elements}
        return cls(node_ids, node_coordinates, element_ids, element_connectivity, element_types)

    @classmethod
    def from_elements(cls, elements: Iterable[AssemblableElement]) -> MeshTopology:
        """Build a :class:`MeshTopology` snapshot directly from a collection of elements.

        For result classes that store their analyzed elements directly
        (e.g. :class:`~femtoolkit.results.analysis_result.AnalysisResult`)
        rather than a :class:`~femtoolkit.mesh.mesh.Mesh`, avoiding the
        need for a caller to separately reconstruct one.

        Args:
            elements: The elements to snapshot; each must expose ``id``
                and ``nodes`` (every :mod:`femtoolkit.mesh` element does).

        Returns:
            A new :class:`MeshTopology`.
        """
        elements = list(elements)
        nodes: dict[int, tuple[float, float, float]] = {}
        for element in elements:
            for node in element.nodes:
                nodes[node.id] = (node.x, node.y, node.z)
        element_ids = tuple(element.id for element in elements)
        element_connectivity = {
            element.id: tuple(node.id for node in element.nodes) for element in elements
        }
        element_types = {element.id: type(element).__name__ for element in elements}
        return cls(tuple(nodes), nodes, element_ids, element_connectivity, element_types)

    def node_coordinate_array(self) -> np.ndarray:
        """Return every node's coordinates as an ``(N, 3)`` array, ordered per :attr:`node_ids`."""
        return np.array(
            [self.node_coordinates[node_id] for node_id in self.node_ids], dtype=float
        )


@dataclass(frozen=True)
class ResultStep:
    """One step's worth of field data: a time instant, a load increment, or a single snapshot.

    Attributes:
        index: This step's position in the result's step sequence
            (``0`` for the first).
        time: The physical time (seconds) for a transient/dynamic step,
            or the load factor (dimensionless, in ``(0, 1]``) for a
            load-stepped nonlinear step. ``0.0`` for a single-step
            static/steady-state result (there is no meaningful "time").
        nodal_fields: Maps a field name (e.g. ``"temperature"``,
            ``"displacement"``) to a ``{node_id: value}`` mapping.
        element_fields: Maps a field name (e.g. ``"stress"``,
            ``"heat_flux"``) to a ``{element_id: value}`` mapping.
    """

    index: int
    time: float
    nodal_fields: dict[str, dict[int, FieldValue]] = field(default_factory=dict)
    element_fields: dict[str, dict[int, FieldValue]] = field(default_factory=dict)

    def nodal_field(self, name: str) -> dict[int, FieldValue]:
        """Return the ``{node_id: value}`` mapping for one nodal field.

        Raises:
            ValidationError: If ``name`` is not a nodal field on this step.
        """
        if name not in self.nodal_fields:
            raise ValidationError(
                f"Step {self.index} has no nodal field {name!r}; available: "
                f"{sorted(self.nodal_fields)}."
            )
        return self.nodal_fields[name]

    def element_field(self, name: str) -> dict[int, FieldValue]:
        """Return the ``{element_id: value}`` mapping for one element field.

        Raises:
            ValidationError: If ``name`` is not an element field on this step.
        """
        if name not in self.element_fields:
            raise ValidationError(
                f"Step {self.index} has no element field {name!r}; available: "
                f"{sorted(self.element_fields)}."
            )
        return self.element_fields[name]

    def nodal_value(self, name: str, node_id: int) -> FieldValue:
        """Return one node's value for one nodal field.

        Raises:
            ValidationError: If ``name`` is not a nodal field on this step.
            EntityNotFoundError: If ``node_id`` has no value for this field.
        """
        values = self.nodal_field(name)
        if node_id not in values:
            raise EntityNotFoundError(f"No {name!r} value for node {node_id} at step {self.index}.")
        return values[node_id]

    def element_value(self, name: str, element_id: int) -> FieldValue:
        """Return one element's value for one element field.

        Raises:
            ValidationError: If ``name`` is not an element field on this step.
            EntityNotFoundError: If ``element_id`` has no value for this field.
        """
        values = self.element_field(name)
        if element_id not in values:
            raise EntityNotFoundError(
                f"No {name!r} value for element {element_id} at step {self.index}."
            )
        return values[element_id]


@dataclass(frozen=True)
class SimulationResult:
    """A solver-independent simulation result: mesh topology plus an ordered sequence of steps.

    Produced by the adapter functions in
    :mod:`femtoolkit.postprocessing.adapters` from any of this toolkit's
    existing result classes -- never constructed directly from solver
    internals, and never itself performs a calculation a solver or field
    calculator has not already done.

    Attributes:
        topology: The mesh's nodes, elements, connectivity, and coordinates.
        steps: Every result step, in order (index 0 first).
        field_units: Optional ``{field_name: unit_string}`` metadata
            (e.g. ``{"temperature": "K"}``), used for labeling in
            visualization/export/reporting.

    Raises:
        ValidationError: If ``steps`` is empty.

    Example:
        >>> result = SimulationResult(topology, steps=(step_0, step_1))
        >>> result.final_step.nodal_value("temperature", node_id=1)
    """

    topology: MeshTopology
    steps: tuple[ResultStep, ...]
    field_units: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that at least one step is present.

        Raises:
            ValidationError: If ``steps`` is empty.
        """
        if not self.steps:
            raise ValidationError("SimulationResult requires at least one step.")

    @property
    def final_step(self) -> ResultStep:
        """The last step (the converged/final state for most analyses)."""
        return self.steps[-1]

    @property
    def num_steps(self) -> int:
        """The number of steps this result contains."""
        return len(self.steps)

    def step(self, index: int) -> ResultStep:
        """Return one step by index (supports negative indexing, e.g. ``-1`` for the last)."""
        return self.steps[index]

    @property
    def times(self) -> np.ndarray:
        """Every step's :attr:`ResultStep.time`, in order."""
        return np.array([step.time for step in self.steps])

    def nodal_history(
        self, field_name: str, node_id: int, component: int | None = None
    ) -> np.ndarray:
        """Return one node's value for one field across every step.

        Args:
            field_name: The nodal field to extract.
            node_id: The node to extract.
            component: If the field is vector-valued, which component to
                extract (e.g. ``0`` for ``ux``). ``None`` for a
                scalar field.

        Returns:
            A NumPy array of shape ``(num_steps,)``.
        """
        values = [step.nodal_value(field_name, node_id) for step in self.steps]
        return _stack_component(values, component)

    def element_history(
        self, field_name: str, element_id: int, component: int | None = None
    ) -> np.ndarray:
        """Return one element's value for one field across every step.

        Args:
            field_name: The element field to extract.
            element_id: The element to extract.
            component: If the field is vector-valued, which component to
                extract. ``None`` for a scalar field.

        Returns:
            A NumPy array of shape ``(num_steps,)``.
        """
        values = [step.element_value(field_name, element_id) for step in self.steps]
        return _stack_component(values, component)


def _stack_component(values: list[FieldValue], component: int | None) -> np.ndarray:
    if component is None:
        return np.array(values, dtype=float)
    return np.array([np.asarray(value)[component] for value in values], dtype=float)
