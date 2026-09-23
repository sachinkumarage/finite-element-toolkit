"""Executes :class:`~femtoolkit.verification.cases.VerificationCase` instances (Version 29).

:class:`VerificationRunner` is the one place that actually calls a
case's :attr:`~femtoolkit.verification.cases.VerificationCase.run`
callable, computes its error metrics, and applies its tolerance --
every concrete benchmark in :mod:`femtoolkit.verification.benchmarks`
only ever *builds* a case; running it always goes through here, so
error handling (a case that cannot be executed becomes
:attr:`~femtoolkit.verification.status.VerificationStatus.NOT_AVAILABLE`,
never an uncaught exception) is implemented exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from femtoolkit.verification.cases import VerificationCase, VerificationResult
from femtoolkit.verification.metrics import (
    absolute_error,
    l2_error,
    relative_error,
    relative_l2_error,
)
from femtoolkit.verification.status import VerificationStatus


def _is_vector(value: float | np.ndarray) -> bool:
    array = np.asarray(value)
    return array.ndim >= 1 and array.size > 1


@dataclass
class VerificationReport:
    """A collection of :class:`~femtoolkit.verification.cases.VerificationResult` entries.

    Attributes:
        results: Every result in the report, in the order they were run.
    """

    results: list[VerificationResult] = field(default_factory=list)

    def count(self, status: VerificationStatus) -> int:
        """Return how many results have the given ``status``."""
        return sum(1 for result in self.results if result.status is status)

    @property
    def passed(self) -> int:
        """Number of results with :attr:`~VerificationStatus.PASS`."""
        return self.count(VerificationStatus.PASS)

    @property
    def failed(self) -> int:
        """Number of results with :attr:`~VerificationStatus.FAIL`."""
        return self.count(VerificationStatus.FAIL)

    @property
    def warnings(self) -> int:
        """Number of results with :attr:`~VerificationStatus.WARNING`."""
        return self.count(VerificationStatus.WARNING)

    @property
    def not_available(self) -> int:
        """Number of results with :attr:`~VerificationStatus.NOT_AVAILABLE`."""
        return self.count(VerificationStatus.NOT_AVAILABLE)

    @property
    def all_passed(self) -> bool:
        """Whether every result in the report has :attr:`~VerificationStatus.PASS`.

        ``True`` for an empty report (vacuously) -- callers checking
        "did everything pass" should also check ``results`` is
        non-empty if an empty report should not count as success.
        """
        return all(result.status is VerificationStatus.PASS for result in self.results)


class VerificationRunner:
    """Runs :class:`~femtoolkit.verification.cases.VerificationCase` instances."""

    def run(self, case: VerificationCase) -> VerificationResult:
        """Execute one verification case.

        Args:
            case: The case to run.

        Returns:
            A :class:`~femtoolkit.verification.cases.VerificationResult`.
            If ``case.run()`` raises, the result has
            :attr:`~femtoolkit.verification.status.VerificationStatus.NOT_AVAILABLE`
            and a message describing the failure -- this method never
            propagates an exception from a case's own FEA model.
        """
        try:
            numerical_value = case.run()
        except Exception as error:  # noqa: BLE001 -- any failure while building/solving a
            # case's own FEA model must degrade to NOT_AVAILABLE, not propagate and abort
            # an entire verification run over one bad case.
            return VerificationResult(
                case_name=case.name,
                description=case.description,
                analysis_type=case.analysis_type,
                quantity=case.quantity,
                reference_value=case.reference_value,
                numerical_value=None,
                absolute_error=None,
                relative_error=None,
                tolerance=case.tolerance,
                status=VerificationStatus.NOT_AVAILABLE,
                message=f"Case could not be executed: {error}",
                units=case.units,
                metadata=case.metadata,
            )

        if _is_vector(case.reference_value) or _is_vector(numerical_value):
            abs_err = l2_error(numerical_value, case.reference_value)
            rel_err = relative_l2_error(numerical_value, case.reference_value)
            reference_norm = float(np.linalg.norm(np.asarray(case.reference_value, dtype=float)))
            satisfied = abs_err <= case.tolerance.allowed_error(reference_norm)
        else:
            abs_err = absolute_error(numerical_value, case.reference_value)
            rel_err = relative_error(numerical_value, case.reference_value)
            satisfied = case.tolerance.is_satisfied(numerical_value, case.reference_value)

        status = VerificationStatus.PASS if satisfied else VerificationStatus.FAIL
        message = (
            f"{case.quantity}: numerical={numerical_value!r}, reference={case.reference_value!r}, "
            f"absolute_error={abs_err:.6e}, relative_error={rel_err:.6e}, "
            f"tolerance(abs={case.tolerance.absolute:.2e}, rel={case.tolerance.relative:.2e}) "
            f"-> {status.value.upper()}"
        )

        return VerificationResult(
            case_name=case.name,
            description=case.description,
            analysis_type=case.analysis_type,
            quantity=case.quantity,
            reference_value=case.reference_value,
            numerical_value=numerical_value,
            absolute_error=abs_err,
            relative_error=rel_err,
            tolerance=case.tolerance,
            status=status,
            message=message,
            units=case.units,
            metadata=case.metadata,
        )

    def run_all(self, cases: list[VerificationCase]) -> VerificationReport:
        """Execute every case in ``cases`` and collect the results.

        Args:
            cases: The cases to run, in order.

        Returns:
            A :class:`VerificationReport` with one
            :class:`~femtoolkit.verification.cases.VerificationResult`
            per case, in the same order.
        """
        return VerificationReport(results=[self.run(case) for case in cases])


__all__ = ["VerificationReport", "VerificationRunner"]
