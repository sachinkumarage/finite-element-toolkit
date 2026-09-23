"""FEA verification: "are we solving the equations correctly?" (Version 29).

Verification is distinct from *validation* (:mod:`femtoolkit.validation`,
"are we solving the correct physical problem?") -- this package never
compares against experimental data, only against known analytical
solutions, mesh-refinement behavior, solver-internal consistency, and
first-principles conservation laws (equilibrium). See
``docs/verification.md`` for the full guide.
"""

from __future__ import annotations

from femtoolkit.verification.benchmarks import (
    axial_bar_cases,
    cantilever_beam_cases,
    cst_patch_case,
    hex8_patch_case,
    quad_patch_case,
    standard_benchmark_suite,
    thermal_conduction_cases,
    truss_cases,
)
from femtoolkit.verification.cases import VerificationCase, VerificationResult
from femtoolkit.verification.checks import (
    EquilibriumCheckResult,
    EquilibriumComponent,
    check_force_equilibrium,
    check_thermal_energy_balance,
)
from femtoolkit.verification.convergence import (
    MeshConvergenceLevel,
    MeshConvergencePoint,
    MeshConvergenceSample,
    MeshConvergenceStudy,
    run_mesh_convergence_study,
)
from femtoolkit.verification.metrics import (
    DEFAULT_EPSILON,
    absolute_error,
    energy_norm_error,
    l2_error,
    relative_error,
    relative_l2_error,
)
from femtoolkit.verification.plots import (
    plot_error_vs_mesh_size,
    plot_mesh_convergence,
    plot_numerical_vs_reference,
    plot_residual_history,
)
from femtoolkit.verification.runner import VerificationReport, VerificationRunner
from femtoolkit.verification.solver_verification import (
    SolverConvergenceRecord,
    compare_solver_solutions,
    solver_convergence_record,
)
from femtoolkit.verification.status import VerificationStatus
from femtoolkit.verification.tolerance import Tolerance

__all__ = [
    "DEFAULT_EPSILON",
    "EquilibriumCheckResult",
    "EquilibriumComponent",
    "MeshConvergenceLevel",
    "MeshConvergencePoint",
    "MeshConvergenceSample",
    "MeshConvergenceStudy",
    "SolverConvergenceRecord",
    "Tolerance",
    "VerificationCase",
    "VerificationReport",
    "VerificationResult",
    "VerificationRunner",
    "VerificationStatus",
    "absolute_error",
    "axial_bar_cases",
    "cantilever_beam_cases",
    "check_force_equilibrium",
    "check_thermal_energy_balance",
    "compare_solver_solutions",
    "cst_patch_case",
    "energy_norm_error",
    "hex8_patch_case",
    "l2_error",
    "plot_error_vs_mesh_size",
    "plot_mesh_convergence",
    "plot_numerical_vs_reference",
    "plot_residual_history",
    "quad_patch_case",
    "relative_error",
    "relative_l2_error",
    "run_mesh_convergence_study",
    "solver_convergence_record",
    "standard_benchmark_suite",
    "thermal_conduction_cases",
    "truss_cases",
]
