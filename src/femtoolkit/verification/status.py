"""The structured outcome vocabulary for verification and validation (Version 29).

A verification or validation result is never reduced to a printed
message or a bare boolean -- every outcome is one of a fixed, small set
of statuses, so a report or a GUI can render, filter, and aggregate
results without parsing text.
"""

from __future__ import annotations

from enum import Enum


class VerificationStatus(Enum):
    """The outcome of one verification or validation comparison.

    Attributes:
        PASS: The numerical result is within the configured tolerance of
            the reference value.
        FAIL: The numerical result exceeds the configured tolerance.
        WARNING: A result exists and was compared, but the comparison
            calls for engineering review rather than an automatic
            pass/fail (e.g. a case explicitly marked as approximate, or
            a mesh-convergence study whose change is not yet monotonic).
        NOT_AVAILABLE: The comparison could not be made because required
            reference data (an analytical solution, a validation
            dataset, solver diagnostics) is unavailable -- never
            silently treated as a pass or a fail.
        NOT_RUN: The case has been defined but has not yet been executed.
    """

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_AVAILABLE = "not_available"
    NOT_RUN = "not_run"

    @property
    def is_favorable(self) -> bool:
        """Whether this status represents an acceptable outcome (``PASS`` only).

        Deliberately excludes ``WARNING``: a warning means "needs
        engineering review," not "acceptable without review."
        """
        return self is VerificationStatus.PASS


__all__ = ["VerificationStatus"]
