"""Engineering equilibrium consistency checks (Version 29).

A converged FEA solve does not, by itself, prove the model was set up
correctly -- a boundary condition applied to the wrong node, or a load
that never actually reached the structure, can still produce a
numerically converged but physically wrong answer. Equilibrium checks
catch exactly this class of error by verifying a conservation law the
underlying formulation *guarantees* at its own converged solution:
applied forces and reactions must balance (structural), and supplied
and removed heat flow must balance (thermal). A mismatch signals a
modeling error, not a solver failure.

Mechanical equilibrium reuses
:class:`~femtoolkit.analysis.dof.DOFMap`'s node-major DOF layout
(``global_index = node_position * dofs_per_node + local_dof``) to group
every global DOF by its physical component (X, Y, RZ, ...) without
needing any new bookkeeping: ``forces[component::dofs_per_node]`` and
``reactions[component::dofs_per_node]`` are exactly every DOF acting in
that direction. Thermal equilibrium reuses the existing Version 21
:func:`~femtoolkit.thermal.thermal_analysis.steady_state_energy_balance`
directly rather than re-deriving it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from femtoolkit.verification.status import VerificationStatus
from femtoolkit.verification.tolerance import Tolerance

if TYPE_CHECKING:
    from femtoolkit.analysis.dof import DOFMap
    from femtoolkit.thermal.thermal_analysis import SteadyStateThermalAnalysis
    from femtoolkit.thermal.thermal_result import SteadyStateThermalResult

_DEFAULT_MECHANICAL_LABELS = ("X", "Y", "RZ")
_DEFAULT_FORCE_TOLERANCE = Tolerance(absolute=1e-6, relative=1e-6)
_DEFAULT_ENERGY_TOLERANCE = Tolerance(absolute=1e-9, relative=1e-6)
"""Combined tolerance, not a bare relative fraction: a purely relative
criterion is meaningless when both the supplied and removed heat flow
are themselves near zero (e.g. a model with only prescribed-temperature
boundaries and no explicit heat source), where a physically negligible
absolute mismatch (floating-point noise, ~1e-12 W) would otherwise
divide by an equally tiny reference value and appear to "fail" by a
large relative margin -- exactly the pitfall
:class:`~femtoolkit.verification.tolerance.Tolerance` exists to avoid."""


@dataclass(frozen=True)
class EquilibriumComponent:
    """One physical component's (e.g. ``"X"``, ``"Y"``, ``"RZ"``) force balance.

    Attributes:
        label: The component's display label.
        applied_total: Sum of applied external loads in this direction.
        reaction_total: Sum of reaction forces in this direction.
        imbalance: ``|applied_total + reaction_total|`` -- zero at exact
            equilibrium.
    """

    label: str
    applied_total: float
    reaction_total: float
    imbalance: float


@dataclass(frozen=True)
class EquilibriumCheckResult:
    """The outcome of one equilibrium consistency check.

    Attributes:
        name: A short check name (e.g. ``"Global force equilibrium"``).
        description: A longer description of what was checked and how.
        components: Per-component balance details (empty for a
            single-quantity check like a thermal energy balance).
        tolerance: The absolute tolerance the maximum imbalance was
            checked against.
        status: :attr:`~femtoolkit.verification.status.VerificationStatus.PASS`
            if every component's imbalance is within ``tolerance``,
            otherwise :attr:`~femtoolkit.verification.status.VerificationStatus.FAIL`.
        message: A short, human-readable summary.
    """

    name: str
    description: str
    components: list[EquilibriumComponent]
    tolerance: Tolerance
    status: VerificationStatus
    message: str
    metadata: dict[str, float] = field(default_factory=dict)


def check_force_equilibrium(
    dof_map: DOFMap,
    forces: np.ndarray,
    reactions: np.ndarray,
    tolerance: Tolerance = _DEFAULT_FORCE_TOLERANCE,
    component_labels: tuple[str, ...] = _DEFAULT_MECHANICAL_LABELS,
) -> EquilibriumCheckResult:
    """Check that applied forces and reactions balance, component by component.

    For a correctly solved, correctly modeled static analysis, the sum
    of every applied external load plus every reaction force in a given
    direction must be zero -- Newton's third law applied to the whole
    structure. Since :class:`~femtoolkit.analysis.dof.DOFMap` lays out
    DOFs node-major (every node's DOFs consecutive, in the same local
    order for every node), ``forces[c::dofs_per_node]`` and
    ``reactions[c::dofs_per_node]`` are exactly every DOF acting in
    component ``c`` (X, Y, RZ, ...), for every node.

    Args:
        dof_map: The DOF map the analysis was solved with (defines
            ``dofs_per_node``, used to group DOFs by component).
        forces: The global applied-force vector, in the same DOF order
            as ``reactions`` (e.g. from
            :func:`~femtoolkit.analysis.system.build_force_vector`).
        reactions: The global reaction vector from
            :attr:`~femtoolkit.results.analysis_result.AnalysisResult.reactions`.
        tolerance: The combined absolute/relative tolerance every
            component's imbalance (checked against zero, so only the
            :attr:`~femtoolkit.verification.tolerance.Tolerance.absolute`
            term is actually used) must be within for the check to
            :attr:`~femtoolkit.verification.status.VerificationStatus.PASS`.
        component_labels: Display labels for each of
            ``dof_map.dofs_per_node`` components, in local-DOF order
            (default ``("X", "Y", "RZ")``, appropriate for a 2D frame
            model; pass ``("X",)`` for an axial bar or ``("X", "Y",
            "Z")`` for a 3D solid with no rotational DOF).

    Returns:
        An :class:`EquilibriumCheckResult` with one
        :class:`EquilibriumComponent` per DOF component.
    """
    dofs_per_node = dof_map.dofs_per_node
    labels = component_labels[:dofs_per_node]

    components = []
    satisfied = True
    for index, label in enumerate(labels):
        applied_total = float(np.sum(forces[index::dofs_per_node]))
        reaction_total = float(np.sum(reactions[index::dofs_per_node]))
        components.append(
            EquilibriumComponent(
                label=label,
                applied_total=applied_total,
                reaction_total=reaction_total,
                imbalance=abs(applied_total + reaction_total),
            )
        )
        # Compare reaction_total against -applied_total (what perfect
        # equilibrium requires) so the tolerance's relative term scales
        # with the actual applied-load magnitude in this component,
        # rather than being compared against a fixed reference of zero.
        satisfied &= tolerance.is_satisfied(reaction_total, -applied_total)

    status = VerificationStatus.PASS if satisfied else VerificationStatus.FAIL
    summary = ", ".join(
        f"{component.label}: imbalance={component.imbalance:.6e}" for component in components
    )
    message = f"Force equilibrium ({summary}) -> {status.value.upper()}"

    return EquilibriumCheckResult(
        name="Global force equilibrium",
        description="Sum of applied external loads and reactions must be zero, per component.",
        components=components,
        tolerance=tolerance,
        status=status,
        message=message,
        metadata={"max_imbalance": max((c.imbalance for c in components), default=0.0)},
    )


def check_thermal_energy_balance(
    analysis: SteadyStateThermalAnalysis,
    result: SteadyStateThermalResult,
    tolerance: Tolerance = _DEFAULT_ENERGY_TOLERANCE,
) -> EquilibriumCheckResult:
    """Check that supplied and removed heat flow balance at steady state.

    Reuses the existing
    :func:`~femtoolkit.thermal.thermal_analysis.steady_state_energy_balance`
    (Version 21) directly: at an exact steady state, heat supplied
    (generation, prescribed flux, net convection/radiation) must equal
    heat removed at the prescribed-temperature boundaries.

    Args:
        analysis: The (already solved) steady-state thermal analysis.
        result: The result returned by ``analysis.solve()``.
        tolerance: The combined absolute/relative tolerance
            ``q_removed`` must be within of ``q_supplied``. A combined
            tolerance (not a bare relative fraction) matters here: a
            model with no explicit heat source has ``q_supplied == 0``,
            where a relative-only criterion would divide a physically
            negligible floating-point mismatch by an equally tiny
            reference and misreport it as a large relative failure.

    Returns:
        An :class:`EquilibriumCheckResult` with no per-component
        breakdown (thermal energy balance is a single scalar quantity).
    """
    from femtoolkit.thermal.thermal_analysis import steady_state_energy_balance

    q_supplied, q_removed = steady_state_energy_balance(analysis, result)
    satisfied = tolerance.is_satisfied(q_removed, q_supplied)
    status = VerificationStatus.PASS if satisfied else VerificationStatus.FAIL
    mismatch = abs(q_supplied - q_removed)
    message = (
        f"Thermal energy balance: q_supplied={q_supplied:.6e} W, q_removed={q_removed:.6e} W, "
        f"mismatch={mismatch:.6e} W -> {status.value.upper()}"
    )

    return EquilibriumCheckResult(
        name="Thermal energy balance",
        description="Heat supplied must equal heat removed at a converged steady state.",
        components=[],
        tolerance=tolerance,
        status=status,
        message=message,
        metadata={"q_supplied": q_supplied, "q_removed": q_removed, "mismatch": mismatch},
    )


__all__ = [
    "EquilibriumCheckResult",
    "EquilibriumComponent",
    "check_force_equilibrium",
    "check_thermal_energy_balance",
]
