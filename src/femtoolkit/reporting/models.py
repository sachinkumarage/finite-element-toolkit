"""The engineering report content model (Version 29).

:class:`EngineeringReport` is a rendering-independent data structure --
it never touches Markdown, HTML, or any other output format itself (see
:mod:`femtoolkit.reporting.renderers` for that). Splitting "what the
report contains" from "how it looks" means a new output format is one
new renderer function, not a rewrite of how reports are built, and
means the report's content can be inspected and tested as plain data.

.. code-block:: text

    SimulationResult, VerificationReport, ValidationResult(s),
    MeshConvergenceStudy, EquilibriumCheckResult(s),
    ReproducibilityMetadata
            |
            v
      EngineeringReport   (this module -- rendering-independent data)
            |
            v
       render_markdown() / render_html()   (femtoolkit.reporting.renderers)

Every section mirrors spec section 12's eighteen-part structure,
grouped into one dataclass field per section (a few closely related
spec sections share one field, e.g. geometry/mesh, where this toolkit's
own mesh summary already covers both). **No section synthesizes an
engineering judgment the caller did not supply** -- :attr:`EngineeringReport.conclusions`
is free text the caller writes; the only thing this module computes
automatically is a factual tally of verification/validation statuses
(:attr:`EngineeringReport.overall_status`), never a statement like "the
structure is safe."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from femtoolkit.verification.status import VerificationStatus

if TYPE_CHECKING:
    from femtoolkit.reporting.metadata import ReproducibilityMetadata
    from femtoolkit.validation.results import ValidationResult
    from femtoolkit.verification.cases import VerificationResult
    from femtoolkit.verification.checks import EquilibriumCheckResult
    from femtoolkit.verification.convergence import MeshConvergenceStudy
    from femtoolkit.verification.solver_verification import SolverConvergenceRecord


@dataclass
class EngineeringReport:
    """The complete content of one engineering simulation report.

    Attributes:
        title: The report title.
        simulation_summary: A short, one-paragraph summary of what was
            simulated and why.
        model_description: A description of the physical model being
            represented.
        geometry_description: A description of the model geometry.
        mesh_summary: Mesh statistics (e.g. ``{"nodes": 120, "elements":
            100, "element_type": "QuadElement2D"}``).
        materials_summary: Material properties used, by name.
        boundary_conditions_summary: A description of the applied
            boundary conditions.
        loads_summary: A description of the applied loads.
        analysis_type: The analysis type that was run.
        solver_configuration: A description of the solver settings used
            (matrix type, solver type, tolerances).
        solver_convergence: The solver's convergence diagnostics, if a
            non-default solver was used (``None`` for the default dense
            direct solve, which has no convergence record).
        verification_results: Every
            :class:`~femtoolkit.verification.cases.VerificationResult`
            relevant to this report.
        validation_results: Every
            :class:`~femtoolkit.validation.results.ValidationResult`
            relevant to this report -- empty if no reference dataset was
            available (never fabricated).
        mesh_convergence: A mesh convergence study, if one was run.
        equilibrium_checks: Every
            :class:`~femtoolkit.verification.checks.EquilibriumCheckResult`
            performed.
        key_results: The simulation's headline engineering results, by
            name (e.g. ``{"Maximum displacement (m)": 0.0012}``).
        warnings: Free-text warnings worth an engineer's attention (mesh
            quality concerns, unconverged studies, ...).
        reproducibility: The
            :class:`~femtoolkit.reporting.metadata.ReproducibilityMetadata`
            for this run.
        conclusions: Free text the caller supplies -- this framework
            never generates an engineering conclusion automatically.
    """

    title: str
    simulation_summary: str
    model_description: str
    geometry_description: str
    mesh_summary: dict[str, int | str]
    materials_summary: dict[str, dict[str, float]]
    boundary_conditions_summary: str
    loads_summary: str
    analysis_type: str
    solver_configuration: str
    reproducibility: ReproducibilityMetadata
    solver_convergence: SolverConvergenceRecord | None = None
    verification_results: list[VerificationResult] = field(default_factory=list)
    validation_results: list[ValidationResult] = field(default_factory=list)
    mesh_convergence: MeshConvergenceStudy | None = None
    equilibrium_checks: list[EquilibriumCheckResult] = field(default_factory=list)
    key_results: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    conclusions: str = ""

    @property
    def overall_status(self) -> VerificationStatus:
        """A single aggregate status across every verification/validation/equilibrium result.

        Computed purely from the structured statuses already present in
        :attr:`verification_results`, :attr:`validation_results`, and
        :attr:`equilibrium_checks` -- a factual tally, not a new
        engineering judgment:

        - :attr:`~femtoolkit.verification.status.VerificationStatus.FAIL`
          if any result failed.
        - :attr:`~femtoolkit.verification.status.VerificationStatus.WARNING`
          if none failed but at least one warrants review.
        - :attr:`~femtoolkit.verification.status.VerificationStatus.NOT_AVAILABLE`
          if no result failed or warned, but at least one could not be
          evaluated.
        - :attr:`~femtoolkit.verification.status.VerificationStatus.PASS`
          if every result passed.
        - :attr:`~femtoolkit.verification.status.VerificationStatus.NOT_RUN`
          if there are no results at all.
        """
        statuses = [result.status for result in self.verification_results]
        statuses += [result.status for result in self.validation_results]
        statuses += [result.status for result in self.equilibrium_checks]
        if self.mesh_convergence is not None:
            statuses.append(self.mesh_convergence.status)

        if not statuses:
            return VerificationStatus.NOT_RUN
        if any(status is VerificationStatus.FAIL for status in statuses):
            return VerificationStatus.FAIL
        if any(status is VerificationStatus.WARNING for status in statuses):
            return VerificationStatus.WARNING
        if any(status is VerificationStatus.NOT_AVAILABLE for status in statuses):
            return VerificationStatus.NOT_AVAILABLE
        return VerificationStatus.PASS


__all__ = ["EngineeringReport"]
