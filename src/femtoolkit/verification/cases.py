"""``VerificationCase`` and ``VerificationResult``: the unit of FEA verification (Version 29).

A :class:`VerificationCase` is a plain, declarative description of one
verification comparison -- what quantity is being checked, against what
reference value, within what tolerance -- paired with a zero-argument
callable that actually builds and solves the FEA model and returns the
numerical value to compare. Separating "what to check" (data) from "how
to compute it" (a callable) keeps every concrete benchmark in
:mod:`femtoolkit.verification.benchmarks` a small, focused function
rather than a subclass, and lets :class:`~femtoolkit.verification.runner.VerificationRunner`
execute any case -- axial bar, truss, beam, thermal conduction, or a
user-defined one -- through the exact same code path.

.. code-block:: text

    VerificationCase
           |
           v
      case.run() -- builds + solves the FEA model, extracts the quantity
           |
           v
      compare numerical vs. reference (femtoolkit.verification.metrics)
           |
           v
      apply tolerance (femtoolkit.verification.tolerance)
           |
           v
      VerificationResult
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from femtoolkit.verification.status import VerificationStatus
from femtoolkit.verification.tolerance import Tolerance


@dataclass(frozen=True)
class VerificationCase:
    """A declarative description of one verification comparison.

    Attributes:
        name: A short, unique, human-readable case name (e.g.
            ``"Axial bar under end load"``).
        description: A longer description of the engineering problem
            and the analytical/reference solution being checked
            against.
        analysis_type: The kind of analysis this case exercises (e.g.
            ``"linear_static"``, ``"thermal_steady_state"``) -- for
            display and report grouping, not dispatch.
        quantity: The physical quantity being verified (e.g.
            ``"Tip displacement"``, ``"Axial stress"``).
        reference_value: The known analytical or trusted reference
            value. A scalar for a single-quantity check, or a
            :class:`numpy.ndarray` for a vector quantity (e.g. every
            nodal displacement in a mesh-convergence study).
        tolerance: The :class:`~femtoolkit.verification.tolerance.Tolerance`
            this case must satisfy to
            :attr:`~femtoolkit.verification.status.VerificationStatus.PASS`.
        run: A zero-argument callable that builds and solves the FEA
            model this case exercises and returns the numerical value
            to compare against :attr:`reference_value` (same
            scalar-or-array shape). Called at most once, by
            :meth:`~femtoolkit.verification.runner.VerificationRunner.run`.
        units: The physical units of :attr:`reference_value`/the
            numerical result (e.g. ``"m"``, ``"Pa"``, ``"K"``), for
            display only.
        metadata: Free-form extra context (e.g. mesh size, element
            type), carried through to :class:`VerificationResult` for
            reporting.
    """

    name: str
    description: str
    analysis_type: str
    quantity: str
    reference_value: float | np.ndarray
    tolerance: Tolerance
    run: Callable[[], float | np.ndarray]
    units: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationResult:
    """The outcome of running one :class:`VerificationCase`.

    Attributes:
        case_name: The originating case's :attr:`VerificationCase.name`.
        description: The originating case's description.
        analysis_type: The originating case's analysis type.
        quantity: The originating case's quantity name.
        reference_value: The reference value the case was checked
            against.
        numerical_value: The FEA result :attr:`VerificationCase.run`
            produced, or ``None`` if the case could not be executed
            (:attr:`status` is
            :attr:`~femtoolkit.verification.status.VerificationStatus.NOT_AVAILABLE`).
        absolute_error: The absolute error, or ``None`` if not computed.
        relative_error: The relative error, or ``None`` if not computed.
        tolerance: The tolerance the case was checked against.
        status: The structured outcome.
        message: A short, human-readable explanation -- always present,
            summarizing the numbers above or explaining why the case
            could not be run.
        units: Physical units, for display only.
        metadata: The originating case's metadata, carried through.
    """

    case_name: str
    description: str
    analysis_type: str
    quantity: str
    reference_value: float | np.ndarray | None
    numerical_value: float | np.ndarray | None
    absolute_error: float | None
    relative_error: float | None
    tolerance: Tolerance | None
    status: VerificationStatus
    message: str
    units: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["VerificationCase", "VerificationResult"]
