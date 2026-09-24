"""`StudyResult`: the collected, serializable outcome of one study (Version 30)."""

from __future__ import annotations

from dataclasses import dataclass, field

from femtoolkit.exceptions import ValidationError
from femtoolkit.runs.models import RunStatus, SimulationRun
from femtoolkit.studies.comparison import DEFAULT_EPSILON, ComparisonResult, compare_runs
from femtoolkit.studies.extractors import Extractor
from femtoolkit.studies.parameter_sweep import ParameterDefinition
from femtoolkit.studies.scenarios import Scenario, get_by_path
from femtoolkit.studies.sensitivity import SensitivityResult, compute_sensitivity_series


@dataclass
class StudyResult:
    """The full, collected outcome of running one
    :class:`~femtoolkit.studies.runner.SimulationStudy`.

    Attributes:
        study_id: A unique identifier for the study.
        study_name: A short, human-readable name.
        base_project_id: The base project's
            :attr:`~femtoolkit.application.project.Project.project_id`.
        parameters: The parameter definitions the study swept, if any
            (empty for a study built from explicit scenarios only).
        scenarios: Every scenario the study executed, in execution
            order.
        runs: Every :class:`~femtoolkit.runs.models.SimulationRun`
            produced, in the same order as ``scenarios``.
    """

    study_id: str
    study_name: str
    base_project_id: str
    parameters: list[ParameterDefinition] = field(default_factory=list)
    scenarios: list[Scenario] = field(default_factory=list)
    runs: list[SimulationRun] = field(default_factory=list)

    @property
    def successful_runs(self) -> list[SimulationRun]:
        """Every run with :attr:`~femtoolkit.runs.models.RunStatus.COMPLETED`."""
        return [run for run in self.runs if run.status == RunStatus.COMPLETED]

    @property
    def failed_runs(self) -> list[SimulationRun]:
        """Every run with :attr:`~femtoolkit.runs.models.RunStatus.FAILED`."""
        return [run for run in self.runs if run.status == RunStatus.FAILED]

    def verification_summary(self) -> dict[str, int]:
        """Tally every run's :attr:`~femtoolkit.runs.models.SimulationRun.verification_status`.

        This tally only reflects what each run's own equilibrium check
        already, genuinely computed (see
        :mod:`femtoolkit.runs.manager`) -- a run that never ran a
        verification check is counted under
        :attr:`~femtoolkit.verification.status.VerificationStatus.NOT_RUN`,
        never assumed to have passed.

        Returns:
            A ``{status_value: count}`` mapping covering every
            :class:`~femtoolkit.verification.status.VerificationStatus`
            member with at least one run.
        """
        tally: dict[str, int] = {}
        for run in self.runs:
            key = run.verification_status.value
            tally[key] = tally.get(key, 0) + 1
        return tally

    def validation_summary(self) -> dict[str, int]:
        """Tally every run's validation status.

        A run with no :attr:`~femtoolkit.runs.models.SimulationRun.validation_result`
        (the default -- validation is never automatic) is counted under
        ``"not_validated"``.

        Returns:
            A ``{status_value_or_"not_validated": count}`` mapping.
        """
        tally: dict[str, int] = {}
        for run in self.runs:
            key = run.validation_result.status.value if run.validation_result else "not_validated"
            tally[key] = tally.get(key, 0) + 1
        return tally

    def compare(
        self, extractor: Extractor, quantity_label: str, epsilon: float = DEFAULT_EPSILON
    ) -> ComparisonResult:
        """Compare ``quantity_label`` across every successful run against the first as baseline.

        A thin, study-scoped wrapper around
        :func:`~femtoolkit.studies.comparison.compare_runs`; see that
        function for the comparison semantics.

        Raises:
            ValidationError: If fewer than two runs succeeded, or the
                quantity is unavailable on any of them.
        """
        return compare_runs(self.successful_runs, extractor, quantity_label, epsilon)

    def sensitivity(
        self, parameter: ParameterDefinition, extractor: Extractor, quantity_label: str
    ) -> list[SensitivityResult]:
        """Compute consecutive pairwise sensitivities of a quantity to one swept parameter.

        Reads each successful run's actual parameter value from its
        immutable :attr:`~femtoolkit.runs.models.SimulationRun.configuration_snapshot`
        (via ``parameter.path``) rather than from ``parameter.values``,
        so the result reflects what each run actually executed with.

        Args:
            parameter: The parameter definition whose swept path to read
                back from each run's configuration snapshot.
            extractor: The result-quantity extractor to observe.
            quantity_label: A human-readable name for the quantity.

        Returns:
            One :class:`~femtoolkit.studies.sensitivity.SensitivityResult`
            per consecutive pair of successful runs, in run order.
        """
        runs = self.successful_runs
        parameter_values = [
            get_by_path(run.configuration_snapshot, parameter.path) for run in runs
        ]
        quantity_values = [extractor(run) for run in runs]
        for run, value in zip(runs, quantity_values, strict=True):
            if value is None:
                raise ValidationError(
                    f"Quantity {quantity_label!r} is not available on run {run.run_id!r}."
                )
        return compute_sensitivity_series(
            parameter_values,
            quantity_values,
            parameter_label=parameter.label,
            quantity_label=quantity_label,
        )
