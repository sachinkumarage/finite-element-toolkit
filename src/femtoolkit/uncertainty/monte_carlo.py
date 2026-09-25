"""Monte Carlo uncertainty studies, built on the Version 30 study infrastructure (Version 31).

.. code-block:: text

    Uncertain Inputs (UncertainParameter list)
            |
            v
    Sampling (femtoolkit.uncertainty.sampling)
            |
            v
    Scenario generation (femtoolkit.studies.scenarios -- one Scenario per sample)
            |
            v
    FEA runs (femtoolkit.studies.runner.StudyRunner / femtoolkit.runs.manager.SimulationRunManager)
            |
            v
    Output collection (MonteCarloResult)
            |
            v
    Statistical analysis (femtoolkit.uncertainty.statistics/confidence/correlation/reliability)

**This module performs no FEA computation and no scenario/run
bookkeeping of its own.** A Monte Carlo sample becomes a
:class:`~femtoolkit.studies.scenarios.Scenario` the exact same way a
Version 30 parameter-sweep value does (``UncertainParameter.path``
follows the same dotted-override-path convention as
:class:`~femtoolkit.studies.parameter_sweep.ParameterDefinition`), and
execution is delegated entirely to
:class:`~femtoolkit.studies.runner.StudyRunner` (or, for
``fail_fast=True``, directly to the same
:class:`~femtoolkit.runs.manager.SimulationRunManager` that
``StudyRunner`` itself uses) -- there is exactly one simulation
execution system in this toolkit, not two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from femtoolkit.application.project import Project
from femtoolkit.exceptions import StudySizeExceededError, ValidationError
from femtoolkit.runs.manager import SimulationRunManager
from femtoolkit.runs.models import RunStatus
from femtoolkit.studies.extractors import Extractor
from femtoolkit.studies.results import StudyResult
from femtoolkit.studies.runner import SimulationStudy, StudyRunner
from femtoolkit.studies.scenarios import Scenario, apply_scenario, validate_unique_scenario_ids
from femtoolkit.uncertainty.parameters import UncertainParameter
from femtoolkit.uncertainty.sampling import (
    RANDOM_METHOD,
    SAMPLING_METHODS,
    SampleSet,
    generate_samples,
)

DEFAULT_MAX_SAMPLES = 1000
"""The default ceiling on requested Monte Carlo samples, mirroring
:data:`~femtoolkit.studies.parameter_sweep.DEFAULT_MAX_SCENARIOS`'s
"stop before execution" role for a deterministic parameter sweep."""

DEFAULT_PERCENTILES: tuple[float, ...] = (5.0, 25.0, 50.0, 75.0, 95.0)


@dataclass
class MonteCarloConfig:
    """The definition of one Monte Carlo uncertainty study, before it is run.

    Attributes:
        study_id: A unique identifier for this study.
        name: A short, human-readable name.
        base_project: The unmodified base project every sample overrides.
        parameters: The uncertain parameters to sample. Must be
            non-empty.
        output_quantities: The named result-quantity extractors (see
            :data:`~femtoolkit.studies.extractors.EXTRACTORS`) to
            collect for every successful run. Must be non-empty.
        n_samples: How many samples to draw.
        seed: A random seed; the same seed reproduces the same sample set.
        method: ``"random"`` or ``"latin_hypercube"``.
        max_samples: The maximum ``n_samples`` this study may request --
            checked before any sampling or execution happens.
        fail_fast: If ``True``, execution stops at the first failed run
            (a solver/validation failure, not a rejected invalid
            sample). If ``False`` (the default), every valid sample is
            executed regardless of earlier failures, matching Version
            30's `StudyRunner` behavior.
        percentiles: Which percentiles (0-100) to report for each
            output quantity.
        confidence_level: The confidence level for the estimated-mean
            confidence interval (e.g. ``0.95``).
    """

    study_id: str
    name: str
    base_project: Project
    parameters: list[UncertainParameter]
    output_quantities: list[str]
    n_samples: int = 100
    seed: int | None = None
    method: str = RANDOM_METHOD
    max_samples: int = DEFAULT_MAX_SAMPLES
    fail_fast: bool = False
    percentiles: tuple[float, ...] = DEFAULT_PERCENTILES
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if not self.parameters:
            raise ValidationError("MonteCarloConfig requires at least one UncertainParameter.")
        if not self.output_quantities:
            raise ValidationError("MonteCarloConfig requires at least one output quantity.")
        if self.method not in SAMPLING_METHODS:
            raise ValidationError(
                f"Unknown sampling method {self.method!r}; expected one of {SAMPLING_METHODS}."
            )
        if self.n_samples < 1:
            raise ValidationError(f"n_samples must be at least 1, got {self.n_samples}.")
        if not (0.0 < self.confidence_level < 1.0):
            raise ValidationError(
                f"confidence_level must lie in (0, 1), got {self.confidence_level!r}."
            )
        if self.n_samples > self.max_samples:
            raise StudySizeExceededError(
                f"Study {self.name!r} requests {self.n_samples} samples, which exceeds "
                f"max_samples={self.max_samples}. Reduce n_samples, or explicitly raise "
                "max_samples if this many simulations is intentional. The study was rejected "
                "before any sample was executed."
            )

    @property
    def estimated_simulation_count(self) -> int:
        """How many FEA solves this study will attempt (one per requested sample)."""
        return self.n_samples


@dataclass(frozen=True)
class InvalidSampleRecord:
    """One sample rejected before execution for violating a parameter's physical bounds.

    Attributes:
        sample_index: The sample's index in the drawn
            :class:`~femtoolkit.uncertainty.sampling.SampleSet`.
        parameter_values: The sample's drawn values, by parameter path.
        reason: A human-readable description of which bound(s) were violated.
    """

    sample_index: int
    parameter_values: dict[str, float]
    reason: str


def _find_physical_violations(parameters: list[UncertainParameter], row: dict[str, float]) -> str:
    violations = []
    for parameter in parameters:
        value = row[parameter.path]
        if not parameter.is_physically_valid(value):
            violations.append(
                f"{parameter.label}={value!r} violates physical bounds "
                f"{parameter.effective_bounds()!r}"
            )
    return "; ".join(violations)


@dataclass
class MonteCarloResult:
    """The collected outcome of running one :class:`MonteCarloConfig`.

    Attributes:
        config: The study configuration that produced this result.
        sample_set: Every drawn sample's input values (valid and
            invalid alike).
        study_result: The underlying Version 30
            :class:`~femtoolkit.studies.results.StudyResult` for every
            sample whose inputs passed physical validation.
        executed_sample_indices: The original sample index each entry
            of ``study_result.runs`` came from, same order and length
            as ``study_result.runs``.
        invalid_samples: Samples rejected before execution for
            violating a physical bound.
        generated_at: ISO-8601 UTC timestamp when this result was produced.
    """

    config: MonteCarloConfig
    sample_set: SampleSet
    study_result: StudyResult
    executed_sample_indices: list[int]
    invalid_samples: list[InvalidSampleRecord] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def n_successful(self) -> int:
        """How many samples completed successfully."""
        return len(self.study_result.successful_runs)

    @property
    def n_failed(self) -> int:
        """How many samples were executed but failed (solver/validation error)."""
        return len(self.study_result.failed_runs)

    @property
    def n_invalid(self) -> int:
        """How many samples were rejected before execution for violating physical bounds."""
        return len(self.invalid_samples)

    def output_values(self, extractor: Extractor) -> np.ndarray:
        """Every successful run's extracted output value, in run order.

        Args:
            extractor: The result-quantity extractor to apply.

        Returns:
            A 1D array of the successful runs' values (runs where the
            extractor returns ``None`` are omitted).
        """
        values = [extractor(run) for run in self.study_result.successful_runs]
        return np.array([value for value in values if value is not None], dtype=float)

    def successful_pairs(
        self, parameter_path: str, extractor: Extractor
    ) -> tuple[np.ndarray, np.ndarray]:
        """Paired (input, output) arrays for successful runs, aligned by original sample index.

        Args:
            parameter_path: Which sampled parameter to pair against the
                output (must be one of ``sample_set.parameter_paths``).
            extractor: The result-quantity extractor to apply.

        Returns:
            ``(x, y)``: ``x`` is that parameter's sampled value and
            ``y`` is the extracted output value, one entry per
            successful run whose output the extractor could compute.
        """
        column_index = self.sample_set.parameter_paths.index(parameter_path)
        xs: list[float] = []
        ys: list[float] = []
        for run, sample_index in zip(
            self.study_result.runs, self.executed_sample_indices, strict=True
        ):
            if run.status != RunStatus.COMPLETED:
                continue
            value = extractor(run)
            if value is None:
                continue
            xs.append(float(self.sample_set.values[sample_index, column_index]))
            ys.append(value)
        return np.array(xs, dtype=float), np.array(ys, dtype=float)


class MonteCarloRunner:
    """Generates uncertain scenarios, executes them, and collects a :class:`MonteCarloResult`."""

    def __init__(
        self,
        study_runner: StudyRunner | None = None,
        run_manager: SimulationRunManager | None = None,
    ) -> None:
        self._run_manager = run_manager or SimulationRunManager()
        self._study_runner = study_runner or StudyRunner(self._run_manager)

    def run(self, config: MonteCarloConfig) -> MonteCarloResult:
        """Sample, execute, and collect the results for one Monte Carlo study.

        Args:
            config: The study configuration (already validated by its
                own ``__post_init__``).

        Returns:
            A :class:`MonteCarloResult`.
        """
        sample_set = generate_samples(
            config.parameters, config.n_samples, config.method, config.seed
        )

        scenarios: list[Scenario] = []
        executed_sample_indices: list[int] = []
        invalid_samples: list[InvalidSampleRecord] = []

        for sample_index in range(config.n_samples):
            row = sample_set.row(sample_index)
            reason = _find_physical_violations(config.parameters, row)
            if reason:
                invalid_samples.append(
                    InvalidSampleRecord(
                        sample_index=sample_index, parameter_values=row, reason=reason
                    )
                )
                continue
            scenarios.append(
                Scenario(
                    scenario_id=f"{config.study_id}-{sample_index}",
                    name=f"{config.name} sample {sample_index}",
                    description=f"Monte Carlo sample {sample_index} of {config.n_samples}.",
                    parameter_overrides=dict(row),
                    metadata={"sample_index": sample_index},
                    tags=["monte-carlo"],
                )
            )
            executed_sample_indices.append(sample_index)

        if config.fail_fast:
            study_result = self._run_fail_fast(config, scenarios)
            n_executed = len(study_result.runs)
            executed_sample_indices = executed_sample_indices[:n_executed]
        else:
            study = SimulationStudy(
                study_id=config.study_id,
                name=config.name,
                base_project=config.base_project,
                scenarios=scenarios,
                max_scenarios=max(config.max_samples, len(scenarios)),
            )
            study_result = self._study_runner.run(study)

        return MonteCarloResult(
            config=config,
            sample_set=sample_set,
            study_result=study_result,
            executed_sample_indices=executed_sample_indices,
            invalid_samples=invalid_samples,
        )

    def _run_fail_fast(self, config: MonteCarloConfig, scenarios: list[Scenario]) -> StudyResult:
        validate_unique_scenario_ids(scenarios)
        runs = []
        for scenario in scenarios:
            project = apply_scenario(config.base_project, scenario)
            run = self._run_manager.execute(project, scenario_id=scenario.scenario_id)
            runs.append(run)
            if run.status == RunStatus.FAILED:
                break
        return StudyResult(
            study_id=config.study_id,
            study_name=config.name,
            base_project_id=config.base_project.project_id,
            parameters=[],
            scenarios=scenarios[: len(runs)],
            runs=runs,
        )


__all__ = [
    "DEFAULT_MAX_SAMPLES",
    "DEFAULT_PERCENTILES",
    "InvalidSampleRecord",
    "MonteCarloConfig",
    "MonteCarloResult",
    "MonteCarloRunner",
]
