"""Pre-solve validation of a project configuration (Version 24).

Section 12 of this version's brief requires every configuration
category to be validated *before* the solver runs, with a clear
explanation when validation fails -- never a raw solver exception, and
never a silent, wrong solve. This module performs only cheap,
structural/physical-range checks on the plain
:class:`~femtoolkit.application.project.Project` data (positive Young's
modulus, a Poisson ratio in the valid isotropic range, a non-empty
mesh, at least one boundary condition, finite load values); it never
constructs a real mesh or material to validate it, so it stays fast and
has no dependency on the solver succeeding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from femtoolkit.application.analysis_types import SUPPORTED_ANALYSIS_TYPES
from femtoolkit.application.project import Project

_REGIONS = ("left", "right", "top", "bottom")
_MECHANICAL_DOFS = ("X", "Y")
_THERMAL_BC_DOFS = ("TEMPERATURE",)
_THERMAL_LOAD_DOFS = ("HEAT_FLUX",)


@dataclass
class ValidationResult:
    """The outcome of validating a project configuration.

    Attributes:
        errors: Every problem found, as a plain human-readable message.
            Empty when the project is valid.
    """

    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Whether the project passed every check (``errors`` is empty)."""
        return not self.errors


def _is_mechanical(analysis_type: str) -> bool:
    return analysis_type in ("linear_static", "nonlinear_static", "thermomechanical")


def _is_thermal(analysis_type: str) -> bool:
    return analysis_type in ("thermal_steady_state", "thermomechanical")


def validate_analysis_type(project: Project) -> list[str]:
    """Check that the project's analysis type is known and currently available."""
    info = SUPPORTED_ANALYSIS_TYPES.get(project.analysis_type)
    if info is None:
        return [f"Unknown analysis type '{project.analysis_type}'."]
    if not info.available:
        return [f"Analysis type '{info.label}' is not yet available. {info.reason}"]
    return []


def validate_material(project: Project) -> list[str]:
    """Check the material's engineering properties for the project's analysis type."""
    errors: list[str] = []
    material = project.material

    if _is_mechanical(project.analysis_type):
        if material.youngs_modulus is None or not math.isfinite(material.youngs_modulus):
            errors.append("Young's modulus must be a finite number.")
        elif material.youngs_modulus <= 0:
            errors.append("Young's modulus must be positive (E > 0).")

        if material.poisson_ratio is None or not math.isfinite(material.poisson_ratio):
            errors.append("Poisson's ratio must be a finite number.")
        elif not (-1.0 < material.poisson_ratio < 0.5):
            errors.append("Poisson's ratio must satisfy -1 < v < 0.5.")

    if _is_thermal(project.analysis_type):
        if material.thermal_conductivity is None or not math.isfinite(
            material.thermal_conductivity
        ):
            errors.append("Thermal conductivity must be a finite number.")
        elif material.thermal_conductivity <= 0:
            errors.append("Thermal conductivity must be positive (k > 0).")

        if material.specific_heat is None or not math.isfinite(material.specific_heat):
            errors.append("Specific heat must be a finite number.")
        elif material.specific_heat <= 0:
            errors.append("Specific heat must be positive (c > 0).")

    if material.density is not None and (
        not math.isfinite(material.density) or material.density <= 0
    ):
        errors.append("Density must be positive when specified.")

    return errors


def validate_mesh(project: Project) -> list[str]:
    """Check the mesh generation parameters."""
    errors: list[str] = []
    mesh = project.mesh

    if not math.isfinite(mesh.width) or mesh.width <= 0:
        errors.append("Mesh width must be positive.")
    if not math.isfinite(mesh.height) or mesh.height <= 0:
        errors.append("Mesh height must be positive.")
    if mesh.nx < 1:
        errors.append("Mesh nx (X subdivisions) must be at least 1.")
    if mesh.ny < 1:
        errors.append("Mesh ny (Y subdivisions) must be at least 1.")
    if mesh.element_type not in ("quad", "cst"):
        errors.append("Mesh element_type must be 'quad' or 'cst'.")
    if _is_mechanical(project.analysis_type) and (
        not math.isfinite(mesh.thickness) or mesh.thickness <= 0
    ):
        errors.append("Element thickness must be positive for a mechanical analysis.")

    return errors


def validate_boundary_conditions(project: Project) -> list[str]:
    """Check that at least one valid boundary condition is configured."""
    errors: list[str] = []
    if not project.boundary_conditions:
        errors.append("No boundary conditions have been defined.")

    allowed_dofs = _THERMAL_BC_DOFS if _is_thermal(project.analysis_type) else _MECHANICAL_DOFS
    for index, bc in enumerate(project.boundary_conditions):
        if bc.region not in _REGIONS:
            errors.append(f"Boundary condition #{index + 1}: unknown region '{bc.region}'.")
        if bc.dof not in allowed_dofs:
            errors.append(
                f"Boundary condition #{index + 1}: dof '{bc.dof}' is not valid for this "
                f"analysis type (expected one of {allowed_dofs})."
            )
        if not math.isfinite(bc.value):
            errors.append(f"Boundary condition #{index + 1}: value must be finite.")

    return errors


def validate_loads(project: Project) -> list[str]:
    """Check that every configured load (if any) is well-formed."""
    errors: list[str] = []
    allowed_dofs = _THERMAL_LOAD_DOFS if _is_thermal(project.analysis_type) else _MECHANICAL_DOFS

    for index, load in enumerate(project.loads):
        if load.region not in _REGIONS:
            errors.append(f"Load #{index + 1}: unknown region '{load.region}'.")
        if load.dof not in allowed_dofs:
            errors.append(
                f"Load #{index + 1}: dof '{load.dof}' is not valid for this analysis type "
                f"(expected one of {allowed_dofs})."
            )
        if not math.isfinite(load.magnitude):
            errors.append(f"Load #{index + 1}: magnitude must be finite.")

    return errors


def validate_solver(project: Project) -> list[str]:
    """Check the solver settings."""
    errors: list[str] = []
    solver = project.solver
    if not math.isfinite(solver.tolerance) or solver.tolerance <= 0:
        errors.append("Solver tolerance must be positive.")
    if solver.max_iterations < 1:
        errors.append("Solver max_iterations must be at least 1.")
    return errors


def validate_project(project: Project) -> ValidationResult:
    """Run every validation category and aggregate the results.

    Args:
        project: The project configuration to validate.

    Returns:
        A :class:`ValidationResult` listing every problem found, empty
        if the project is ready to run.
    """
    errors = list(validate_analysis_type(project))
    if not errors:
        # Every other check assumes a known, available analysis type.
        errors += validate_material(project)
        errors += validate_mesh(project)
        errors += validate_boundary_conditions(project)
        errors += validate_loads(project)
    errors += validate_solver(project)
    return ValidationResult(errors=errors)
