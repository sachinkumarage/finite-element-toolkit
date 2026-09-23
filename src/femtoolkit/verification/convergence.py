"""Mesh convergence studies (Version 29).

A mesh convergence study asks a different question than a benchmark
comparison against a known analytical value: "as the mesh is refined,
does the result quantity settle down to a stable value?" This is
essential verification even when no analytical solution exists at all
-- a converging sequence of results across coarse/medium/fine/finer
meshes is itself evidence the discretization is behaving correctly, and
a non-converging or erratic sequence is a red flag regardless of
whether any reference value is available.

.. code-block:: text

    Coarse -> Medium -> Fine -> Finer
       |         |        |       |
       v         v        v       v
    result    result   result  result
       \\_________|________|_______/
                  relative change between
                  consecutive levels

Convergence is **not** assumed to be monotonic -- a result can
legitimately oscillate while still settling toward a stable value (this
is common, for example, near a stress concentration with a coarse
mesh), so this module reports the relative change at every level rather
than asserting the sequence is monotonically decreasing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from femtoolkit.verification.metrics import DEFAULT_EPSILON
from femtoolkit.verification.status import VerificationStatus


@dataclass(frozen=True)
class MeshConvergenceSample:
    """One mesh resolution's measured model size and result quantity.

    Attributes:
        num_nodes: Number of mesh nodes at this resolution.
        num_elements: Number of mesh elements at this resolution.
        num_dofs: Number of degrees of freedom at this resolution.
        result_value: The selected result quantity's value at this
            resolution (e.g. a tip displacement, a maximum stress).
    """

    num_nodes: int
    num_elements: int
    num_dofs: int
    result_value: float


@dataclass(frozen=True)
class MeshConvergenceLevel:
    """One mesh resolution to include in a convergence study.

    Attributes:
        label: A short, human-readable resolution label (e.g.
            ``"Coarse"``, ``"Medium"``, ``"Fine"``, ``"Finer"``).
        mesh_size: A representative element size for this level (e.g.
            the target edge length, in meters) -- used only for
            plotting/display, smaller values expected to indicate a
            finer mesh.
        run: A zero-argument callable that builds and solves the FEA
            model at this resolution and returns a
            :class:`MeshConvergenceSample`.
    """

    label: str
    mesh_size: float
    run: Callable[[], MeshConvergenceSample]


@dataclass(frozen=True)
class MeshConvergencePoint:
    """One resolved level's recorded data and its change from the previous level.

    Attributes:
        label: The originating level's label.
        mesh_size: The originating level's mesh size.
        num_nodes: Node count at this resolution.
        num_elements: Element count at this resolution.
        num_dofs: DOF count at this resolution.
        result_value: The result quantity's value at this resolution.
        relative_change: ``|Q_i - Q_{i-1}| / max(|Q_i|, eps)``, or
            ``None`` for the first (coarsest) level, which has no
            predecessor to compare against.
    """

    label: str
    mesh_size: float
    num_nodes: int
    num_elements: int
    num_dofs: int
    result_value: float
    relative_change: float | None


@dataclass
class MeshConvergenceStudy:
    """The complete recorded history of a mesh convergence study.

    Attributes:
        quantity: The result quantity being tracked (e.g. ``"Tip
            displacement"``), for display.
        tolerance: The relative-change tolerance
            :attr:`status` was evaluated against.
        points: Every resolved level, coarsest first, in the order
            they were run.
        status: :attr:`~femtoolkit.verification.status.VerificationStatus.PASS`
            if the final level's relative change is within
            ``tolerance``; :attr:`~femtoolkit.verification.status.VerificationStatus.WARNING`
            if it is not (the study does not conclude the model is
            wrong -- only that convergence was not demonstrated within
            the tested mesh levels, which calls for engineering review,
            e.g. testing a finer mesh);
            :attr:`~femtoolkit.verification.status.VerificationStatus.NOT_AVAILABLE`
            if fewer than two levels were run (a relative change needs
            at least two points).
    """

    quantity: str
    tolerance: float
    points: list[MeshConvergencePoint] = field(default_factory=list)
    status: VerificationStatus = VerificationStatus.NOT_RUN

    @property
    def final_relative_change(self) -> float | None:
        """The last recorded level's relative change, or ``None`` if unavailable."""
        return self.points[-1].relative_change if self.points else None


def run_mesh_convergence_study(
    quantity: str,
    levels: list[MeshConvergenceLevel],
    tolerance: float = 1e-3,
    epsilon: float = DEFAULT_EPSILON,
) -> MeshConvergenceStudy:
    """Run every level of a mesh convergence study and record its history.

    Args:
        quantity: The result quantity being tracked, for display.
        levels: The mesh resolutions to run, ordered coarsest to finest
            (the order results are recorded in; relative change is
            always computed against the immediately preceding entry in
            this list, regardless of ``mesh_size`` ordering).
        tolerance: The relative-change tolerance the final level must
            satisfy for :attr:`MeshConvergenceStudy.status` to be
            :attr:`~femtoolkit.verification.status.VerificationStatus.PASS`.
        epsilon: Floor for the relative-change denominator (see
            :mod:`femtoolkit.verification.metrics`).

    Returns:
        A :class:`MeshConvergenceStudy` with one
        :class:`MeshConvergencePoint` per level.
    """
    points: list[MeshConvergencePoint] = []
    previous_value: float | None = None

    for level in levels:
        sample = level.run()
        relative_change = (
            None
            if previous_value is None
            else abs(sample.result_value - previous_value) / max(abs(sample.result_value), epsilon)
        )
        points.append(
            MeshConvergencePoint(
                label=level.label,
                mesh_size=level.mesh_size,
                num_nodes=sample.num_nodes,
                num_elements=sample.num_elements,
                num_dofs=sample.num_dofs,
                result_value=sample.result_value,
                relative_change=relative_change,
            )
        )
        previous_value = sample.result_value

    if len(points) < 2:
        status = VerificationStatus.NOT_AVAILABLE
    elif points[-1].relative_change is not None and points[-1].relative_change < tolerance:
        status = VerificationStatus.PASS
    else:
        status = VerificationStatus.WARNING

    return MeshConvergenceStudy(
        quantity=quantity, tolerance=tolerance, points=points, status=status
    )


__all__ = [
    "MeshConvergenceLevel",
    "MeshConvergencePoint",
    "MeshConvergenceSample",
    "MeshConvergenceStudy",
    "run_mesh_convergence_study",
]
