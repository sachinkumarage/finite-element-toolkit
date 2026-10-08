"""Surrogate-assisted adaptive optimization (Version 36).

**Mathematical foundation.** For a design vector :math:`x = (x_1, x_2, \\ldots, x_n)`,
the expensive high-fidelity simulation provides :math:`y = f(x)` (a
:class:`~femtoolkit.application.project.Project` solved by this toolkit's existing FEA
pipeline). The Version 35 surrogate approximates :math:`\\hat f(x) \\approx f(x)`. The
underlying optimization problem is

.. math::

    \\min_x f(x) \\quad \\text{s.t.} \\quad g_i(x) \\leq 0, \\quad x_{min} \\leq x \\leq x_{max}

-- but every candidate this package proposes is scored against :math:`\\hat f`, never
:math:`f`, until it is explicitly verified. **The surrogate is an acceleration tool,
not a replacement**: :math:`f` remains the engineering reference throughout.

.. code-block:: text

    Initial Engineering Samples
            |
            v
    High-Fidelity FEA
            |
            v
    Surrogate Model
            |
            v
    Optimization / Candidate Search
            |
            v
    Candidate Designs
            |
            v
    High-Fidelity Verification
            |
            v
    Add New Samples -> Retrain Surrogate -> Repeat

Built entirely on top of the existing Version 30 simulation workflows
(:mod:`femtoolkit.studies`, :mod:`femtoolkit.runs`), Version 31 uncertainty
infrastructure (:mod:`femtoolkit.uncertainty`, usable through a
:func:`~femtoolkit.optimization.robust.robust_objective_statistic`-built
:class:`~femtoolkit.optimization.objectives.Objective` exactly as Version 33
optimization already supports), Version 33 optimization interfaces
(:mod:`femtoolkit.optimization`), Version 34 parallel execution
(:mod:`femtoolkit.orchestration`), and Version 35 surrogate/reduced-order modeling
(:mod:`femtoolkit.surrogate`) -- no scaling, regression, validation, dataset, or
evaluator logic is duplicated here.

**Explicit scope exclusions (this version).** No multi-fidelity modeling, co-kriging,
Gaussian processes, deep learning, neural networks, topology/shape/adjoint
optimization, reinforcement learning, distributed/GPU/cloud execution, digital twins,
formal reliability methods (FORM/SORM), or advanced Bayesian optimization -- see
``docs/releases/v36.0.0.md`` for the full scope boundary and the Version 37 preview.
"""

from __future__ import annotations

from femtoolkit.adaptive.candidates import clip_to_bounds, generate_candidate_pool
from femtoolkit.adaptive.refinement import (
    RefinementConfig,
    RefinementStepResult,
    SurrogateAcceptanceState,
    prediction_agreement,
    run_refinement_step,
)
from femtoolkit.adaptive.results import AdaptiveStudyResult
from femtoolkit.adaptive.sampling import (
    CandidateScore,
    SamplingStrategy,
    design_variable_bounds,
    error_score,
    exploration_score,
    normalized_distance,
    rank_candidates,
)
from femtoolkit.adaptive.study import AdaptiveStudy, generate_initial_samples
from femtoolkit.adaptive.trust_region import TrustRegion

__all__ = [
    "AdaptiveStudy",
    "AdaptiveStudyResult",
    "CandidateScore",
    "RefinementConfig",
    "RefinementStepResult",
    "SamplingStrategy",
    "SurrogateAcceptanceState",
    "TrustRegion",
    "clip_to_bounds",
    "design_variable_bounds",
    "error_score",
    "exploration_score",
    "generate_candidate_pool",
    "generate_initial_samples",
    "normalized_distance",
    "prediction_agreement",
    "rank_candidates",
    "run_refinement_step",
]
