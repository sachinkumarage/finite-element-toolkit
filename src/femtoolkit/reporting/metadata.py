"""Reproducibility metadata: how a result was produced (Version 29).

The purpose of :class:`ReproducibilityMetadata` is stated directly in
spec section 11: "allow another engineer to understand how a result was
produced." It is a plain, serializable data record -- no behavior, no
solving -- collected once after a simulation has run and attached to an
:class:`~femtoolkit.reporting.models.EngineeringReport`.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from typing import Any

_TRACKED_DEPENDENCIES = ("numpy", "scipy", "matplotlib")
"""Dependencies whose installed version is recorded when practical (spec
section 11: "dependency versions where practical") -- the toolkit's
actual required/core dependencies, not every transitively installed
package."""


def collect_dependency_versions(
    package_names: tuple[str, ...] = _TRACKED_DEPENDENCIES,
) -> dict[str, str]:
    """Look up the installed version of each named package.

    Args:
        package_names: Distribution names to look up (as installed via
            pip, not necessarily the import name).

    Returns:
        A ``{package_name: version}`` mapping. A package that cannot be
        found (not installed, or looked up under a different
        distribution name) is simply omitted rather than raising --
        reproducibility metadata should degrade gracefully, never abort
        a report because one optional dependency's version could not be
        determined.
    """
    versions: dict[str, str] = {}
    for name in package_names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


@dataclass(frozen=True)
class ReproducibilityMetadata:
    """A structured record of how one simulation result was produced.

    Attributes:
        toolkit_version: This toolkit's version (``femtoolkit.config.__version__``).
        python_version: The running Python interpreter's version.
        platform_description: A short OS/platform description.
        dependency_versions: Installed versions of the toolkit's core
            dependencies (see :func:`collect_dependency_versions`).
        model_name: A human-readable name for the model this metadata
            describes.
        analysis_type: The analysis type (e.g. ``"linear_static"``).
        mesh_statistics: Node/element/DOF counts and similar mesh-size
            facts.
        element_types: The distinct element type names used in the mesh.
        material_properties: The material properties used, by name.
        boundary_conditions_summary: A short description of the
            boundary conditions applied.
        loads_summary: A short description of the loads applied.
        solver_name: The solver strategy that actually ran.
        preconditioner: The preconditioner used, if any (``None`` --
            this toolkit does not yet implement solver preconditioning;
            see the Version 28 preview).
        solver_tolerances: The solver's configured tolerances, by name
            (e.g. ``{"tolerance": 1e-8, "max_iterations": 1000}``).
        degrees_of_freedom: Total DOF count of the solved system.
        execution_mode: ``"serial"`` or ``"parallel"`` (Version 27
            element-computation execution mode), if recorded.
        random_seed: The random seed used, if any randomized process was
            involved in producing this result (``None`` if the
            simulation is fully deterministic, which every analysis in
            this toolkit is unless a caller has introduced randomness of
            their own).
        generated_at: ISO-8601 UTC timestamp when this metadata was
            collected.
        extra: Free-form additional configuration parameters relevant to
            reproducing the result.
    """

    toolkit_version: str
    python_version: str
    platform_description: str
    dependency_versions: dict[str, str]
    model_name: str
    analysis_type: str
    mesh_statistics: dict[str, int]
    element_types: list[str]
    material_properties: dict[str, float]
    boundary_conditions_summary: str
    loads_summary: str
    solver_name: str
    preconditioner: str | None
    solver_tolerances: dict[str, float]
    degrees_of_freedom: int | None
    execution_mode: str | None
    random_seed: int | None = None
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a plain, JSON-serializable representation of this metadata."""
        return asdict(self)


def collect_reproducibility_metadata(
    model_name: str,
    analysis_type: str,
    mesh_statistics: dict[str, int],
    element_types: list[str],
    material_properties: dict[str, float],
    boundary_conditions_summary: str,
    loads_summary: str,
    solver_name: str = "Dense Direct",
    solver_tolerances: dict[str, float] | None = None,
    degrees_of_freedom: int | None = None,
    execution_mode: str | None = "serial",
    random_seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> ReproducibilityMetadata:
    """Collect a :class:`ReproducibilityMetadata` record for the current run.

    Args:
        model_name: A human-readable name for the model.
        analysis_type: The analysis type.
        mesh_statistics: Node/element/DOF counts and similar facts.
        element_types: The distinct element type names used.
        material_properties: The material properties used, by name.
        boundary_conditions_summary: A short boundary-condition summary.
        loads_summary: A short load summary.
        solver_name: The solver strategy that ran.
        solver_tolerances: The solver's configured tolerances, if any.
        degrees_of_freedom: Total DOF count.
        execution_mode: The Version 27 element-computation execution
            mode, if applicable.
        random_seed: The random seed used, if any.
        extra: Free-form additional configuration parameters.

    Returns:
        A :class:`ReproducibilityMetadata` populated with the current
        toolkit/Python/dependency versions plus the given model context.
    """
    from femtoolkit.config import __version__ as toolkit_version

    return ReproducibilityMetadata(
        toolkit_version=toolkit_version,
        python_version=sys.version.split()[0],
        platform_description=platform.platform(),
        dependency_versions=collect_dependency_versions(),
        model_name=model_name,
        analysis_type=analysis_type,
        mesh_statistics=dict(mesh_statistics),
        element_types=list(element_types),
        material_properties=dict(material_properties),
        boundary_conditions_summary=boundary_conditions_summary,
        loads_summary=loads_summary,
        solver_name=solver_name,
        preconditioner=None,
        solver_tolerances=dict(solver_tolerances or {}),
        degrees_of_freedom=degrees_of_freedom,
        execution_mode=execution_mode,
        random_seed=random_seed,
        extra=dict(extra or {}),
    )


__all__ = [
    "ReproducibilityMetadata",
    "collect_dependency_versions",
    "collect_reproducibility_metadata",
]
