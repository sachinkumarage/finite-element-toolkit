"""The serializable engineering project model (Version 24).

An engineering project is everything a user configures through the GUI
*before* a simulation runs: which analysis type, which material, what
mesh, which boundary conditions and loads, and what solver settings --
"Project -> Model -> Material -> Mesh -> Boundary Conditions -> Loads ->
Solver" from the GUI workflow this version implements.

This module intentionally stores only **plain, JSON-serializable data**
(strings, numbers, lists of small dataclasses) -- never a live
:class:`~femtoolkit.mesh.mesh.Mesh`, material, or analysis object. Those
are *derived* from a :class:`Project` on demand by
:mod:`femtoolkit.application.model_service`, exactly once, right before
a simulation runs. Serializing a `Project` therefore never needs to
pickle a solver object or a numpy array -- only the configuration that
describes how to build one.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from femtoolkit.exceptions import ValidationError

PROJECT_FORMAT_VERSION = 1
"""Schema version written into every saved project file, so a future
version of this toolkit can detect and migrate an older project file
rather than silently misreading it."""


@dataclass
class MaterialConfig:
    """The material properties a project is configured with.

    Every field is optional because different analysis types need
    different subsets (a purely thermal analysis has no
    ``youngs_modulus``; a purely mechanical one has no
    ``thermal_conductivity``) -- see
    :mod:`femtoolkit.application.validation` for which fields are
    required for a given analysis type.

    Attributes:
        name: A human-readable material name (e.g. ``"Structural Steel"``).
        youngs_modulus: Young's modulus, in pascals.
        poisson_ratio: Poisson's ratio (dimensionless).
        density: Mass density, in kg/m^3.
        thermal_conductivity: Thermal conductivity, in W/(m*K).
        specific_heat: Specific heat capacity, in J/(kg*K).
    """

    name: str = "Custom Material"
    youngs_modulus: float | None = None
    poisson_ratio: float | None = None
    density: float | None = None
    thermal_conductivity: float | None = None
    specific_heat: float | None = None


@dataclass
class MeshConfig:
    """Structured 2D mesh generation parameters.

    Reuses :func:`~femtoolkit.mesh.generator.create_quad_mesh`/
    :func:`~femtoolkit.mesh.generator.create_triangular_mesh` (Version 8)
    over a :class:`~femtoolkit.geometry.rectangle.Rectangle` (Version 9)
    domain -- this version does not add a new mesh generator.

    Attributes:
        width: Domain width (X extent), in meters.
        height: Domain height (Y extent), in meters.
        nx: Subdivisions along X.
        ny: Subdivisions along Y.
        element_type: ``"quad"`` (Q4) or ``"cst"`` (constant-strain triangle).
        thickness: Element thickness, in meters (mechanical analyses only).
        refinement_passes: How many uniform refinement passes (Version
            25, :func:`~femtoolkit.mesh.refinement.refine_uniform`) to
            apply to the generated mesh before it is handed to a
            solver. ``0`` (the default) reproduces every Version 24
            project's exact prior behavior unchanged.
    """

    width: float = 2.0
    height: float = 0.4
    nx: int = 10
    ny: int = 2
    element_type: str = "quad"
    thickness: float = 0.02
    refinement_passes: int = 0


@dataclass
class BoundaryConditionConfig:
    """One essential boundary condition applied to every node on a named region.

    Attributes:
        region: One of ``"left"``/``"right"``/``"top"``/``"bottom"``
            (:data:`~femtoolkit.geometry.rectangle.BOUNDARY_NAMES`).
        dof: ``"X"``/``"Y"`` (a prescribed displacement, mechanical
            analyses) or ``"TEMPERATURE"`` (a prescribed temperature,
            thermal analyses).
        value: The prescribed value, in meters (``"X"``/``"Y"``) or
            kelvin (``"TEMPERATURE"``).
    """

    region: str = "left"
    dof: str = "X"
    value: float = 0.0


@dataclass
class LoadConfig:
    """One nodal load applied to every node on a named region.

    Attributes:
        region: One of ``"left"``/``"right"``/``"top"``/``"bottom"``.
        dof: ``"X"``/``"Y"`` (a nodal force, mechanical analyses) or
            ``"HEAT_FLUX"`` (a prescribed nodal heat flow, thermal
            analyses).
        magnitude: The applied value, in newtons (``"X"``/``"Y"``) or
            watts (``"HEAT_FLUX"``), applied to **every** node on the
            region (not distributed as a single resultant).
    """

    region: str = "right"
    dof: str = "X"
    magnitude: float = 0.0


@dataclass
class SolverConfig:
    """Solver settings exposed to the user.

    Attributes:
        matrix_type: ``"dense"`` or ``"sparse"`` (Version 26) -- which
            matrix representation the global stiffness/conductivity
            matrix is assembled in.
        solver_type: ``"direct"`` or ``"conjugate_gradient"`` (Version
            26). ``"conjugate_gradient"`` requires ``matrix_type ==
            "sparse"``; see
            :func:`~femtoolkit.application.model_service.ModelService.build_solver`.
        tolerance: Convergence tolerance, used only by
            ``solver_type == "conjugate_gradient"``.
        max_iterations: Maximum solver iterations, used only by
            ``solver_type == "conjugate_gradient"``.
    """

    matrix_type: str = "dense"
    solver_type: str = "direct"
    tolerance: float = 1e-6
    max_iterations: int = 1000


@dataclass
class ExecutionSettingsConfig:
    """Element-computation execution settings exposed to the user (Version 27).

    Attributes:
        mode: ``"serial"`` (the default) or ``"parallel"`` -- how
            per-element stiffness/conductivity matrices are computed;
            see :class:`~femtoolkit.execution.config.ExecutionConfig`.
        workers: Number of workers when ``mode == "parallel"``. ``None``
            (the default) resolves automatically; see
            :func:`~femtoolkit.execution.config.resolve_workers`.
        batch_size: ``"auto"`` (the default) or a positive integer
            number of elements per worker task; see
            :func:`~femtoolkit.execution.executor.resolve_chunk_size`.
    """

    mode: str = "serial"
    workers: int | None = None
    batch_size: str | int = "auto"


@dataclass
class Project:
    """A complete, serializable engineering project configuration.

    Attributes:
        name: The project's display name.
        analysis_type: A key from
            :data:`~femtoolkit.application.analysis_types.SUPPORTED_ANALYSIS_TYPES`.
        material: The configured material.
        mesh: The configured mesh generation parameters.
        boundary_conditions: Every configured boundary condition.
        loads: Every configured load.
        solver: The configured solver settings.
        execution: The configured element-computation execution settings
            (Version 27).
        created_at: ISO-8601 UTC timestamp set at creation time.
        format_version: The schema version this project was written with.
        project_id: A stable identifier for this project, generated once
            at creation time and preserved across save/load round-trips
            (Version 30). Used to trace a
            :class:`~femtoolkit.runs.models.SimulationRun` back to the
            project it was executed from -- this project *is* the
            "simulation project" concept referenced by that version,
            rather than a separate duplicated model.
    """

    name: str = "Untitled Project"
    analysis_type: str = "linear_static"
    material: MaterialConfig = field(default_factory=MaterialConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    boundary_conditions: list[BoundaryConditionConfig] = field(default_factory=list)
    loads: list[LoadConfig] = field(default_factory=list)
    solver: SolverConfig = field(default_factory=SolverConfig)
    execution: ExecutionSettingsConfig = field(default_factory=ExecutionSettingsConfig)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    format_version: int = PROJECT_FORMAT_VERSION
    project_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        """Return a plain, JSON-serializable representation of this project."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Project:
        """Reconstruct a :class:`Project` from :meth:`to_dict`'s output.

        Args:
            data: A dictionary shaped like :meth:`to_dict`'s output.
                Unknown top-level keys are ignored, so a project saved
                by a future version with extra fields can still be
                partially read.

        Returns:
            A new :class:`Project`.

        Raises:
            ValidationError: If ``data["format_version"]`` is newer than
                this toolkit's own :data:`PROJECT_FORMAT_VERSION` (a
                project saved by a future version this toolkit does not
                yet know how to read -- no migration framework exists
                yet, so this is reported rather than silently
                misinterpreted).
        """
        format_version = data.get("format_version", PROJECT_FORMAT_VERSION)
        if format_version > PROJECT_FORMAT_VERSION:
            raise ValidationError(
                f"Project format_version {format_version} is newer than this toolkit "
                f"supports (format_version {PROJECT_FORMAT_VERSION}); no migration path "
                "exists yet for a project saved by a future version."
            )
        return cls(
            name=data.get("name", "Untitled Project"),
            analysis_type=data.get("analysis_type", "linear_static"),
            material=MaterialConfig(**data.get("material", {})),
            mesh=MeshConfig(**data.get("mesh", {})),
            boundary_conditions=[
                BoundaryConditionConfig(**bc) for bc in data.get("boundary_conditions", [])
            ],
            loads=[LoadConfig(**load) for load in data.get("loads", [])],
            solver=SolverConfig(**data.get("solver", {})),
            execution=ExecutionSettingsConfig(**data.get("execution", {})),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            format_version=format_version,
            project_id=data.get("project_id") or str(uuid.uuid4()),
        )
