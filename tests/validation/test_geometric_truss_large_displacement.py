"""Validation: large-displacement truss vs. the linear solution (spec section 16, 28).

Uses a classic **shallow (von Mises) truss** -- two members meeting at a
low apex, spanning a much larger horizontal distance -- the standard
textbook structure for demonstrating geometric nonlinearity, because a
small vertical apex load produces a *large* rotation of both members even
though nodal displacements stay modest, exactly the regime where the
small-strain linear `B` matrix breaks down.

**Snap-through foundation (spec section 28).** A shallow truss loaded
further is the standard example that begins to exhibit snap-through: as
the apex is pushed down toward (and potentially past) the line connecting
its two supports, the structure's stiffness in the load direction drops
toward zero and the Newton-Raphson iteration count for a fixed load
increment grows sharply -- the honest, load-controlled signature of
approaching (or reaching) a limit point, without needing (and this project
does not implement) the arc-length/Riks method that would be needed to
trace *past* a true limit point. This test demonstrates and documents that
signature rather than attempting to solve through an actual snap.
"""

import numpy as np
import pytest

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.materials import Material, SaintVenantKirchhoff1D
from femtoolkit.mesh import Mesh, Node, TrussElement2D
from femtoolkit.sections import CrossSection

X = TranslationDOF.X
Y = TranslationDOF.Y

RISE = 0.2
SPAN = 5.0
YOUNGS_MODULUS = 200e9
AREA = 0.0005


def _build_shallow_truss_mesh() -> tuple[Mesh, TrussElement2D, TrussElement2D]:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=SPAN / 2, y=RISE, z=0.0)  # apex
    node_3 = Node(id=3, x=SPAN, y=0.0, z=0.0)
    placeholder = Material(
        name="steel", density=7850.0, youngs_modulus=YOUNGS_MODULUS, poissons_ratio=0.3
    )
    section = CrossSection(area=AREA)
    truss_1 = TrussElement2D(
        id=1, nodes=(node_1, node_2), material=placeholder, cross_section=section
    )
    truss_2 = TrussElement2D(
        id=2, nodes=(node_2, node_3), material=placeholder, cross_section=section
    )

    mesh = Mesh()
    for node in (node_1, node_2, node_3):
        mesh.add_node(node)
    mesh.add_element(truss_1)
    mesh.add_element(truss_2)
    return mesh, truss_1, truss_2


def test_nonlinear_solution_diverges_significantly_from_linear() -> None:
    """A moderate apex load already produces a visibly different nonlinear result."""
    mesh, truss_1, truss_2 = _build_shallow_truss_mesh()
    apex_load = -2.0e4

    linear_analysis = StaticLinearAnalysis(mesh)
    linear_analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    linear_analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    linear_analysis.add_boundary_condition(BoundaryCondition(3, X, 0.0))
    linear_analysis.add_boundary_condition(BoundaryCondition(3, Y, 0.0))
    linear_analysis.add_load(NodalLoad(2, Y, apex_load))
    linear_result = linear_analysis.solve()
    linear_uy = linear_result.displacement(2, Y)

    material = SaintVenantKirchhoff1D(youngs_modulus=YOUNGS_MODULUS)
    settings = NonlinearSolverSettings(load_steps=20, tolerance=1e-9, max_iterations=60)
    nonlinear_analysis = NonlinearAnalysis(
        mesh, {truss_1.id: material, truss_2.id: material}, settings, geometric_nonlinearity=True
    )
    nonlinear_analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    nonlinear_analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    nonlinear_analysis.add_boundary_condition(BoundaryCondition(3, X, 0.0))
    nonlinear_analysis.add_boundary_condition(BoundaryCondition(3, Y, 0.0))
    nonlinear_analysis.add_load(NodalLoad(2, Y, apex_load))
    nonlinear_result = nonlinear_analysis.solve()

    assert nonlinear_result.converged
    nonlinear_uy = nonlinear_result.displacement(2, Y)

    # The whole point of the shallow-truss test case: nonlinear and linear
    # solutions must differ substantially (linear analysis badly
    # underestimates deflection in this regime).
    assert abs(nonlinear_uy / linear_uy) > 2.0


def test_load_stepping_iteration_count_grows_as_apex_approaches_the_span_line() -> None:
    """As the load increases toward the shallow truss's limit point, Newton-Raphson
    needs visibly more iterations per step -- the honest, load-controlled signature
    of approaching snap-through (spec section 28's explicitly sanctioned alternative
    to implementing an arc-length method)."""
    mesh, truss_1, truss_2 = _build_shallow_truss_mesh()
    material = SaintVenantKirchhoff1D(youngs_modulus=YOUNGS_MODULUS)
    settings = NonlinearSolverSettings(load_steps=20, tolerance=1e-9, max_iterations=80)
    analysis = NonlinearAnalysis(
        mesh, {truss_1.id: material, truss_2.id: material}, settings, geometric_nonlinearity=True
    )
    analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(3, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(3, Y, 0.0))
    analysis.add_load(NodalLoad(2, Y, -2.0e4))  # close to this truss's limit point

    result = analysis.solve()
    assert result.converged

    iteration_counts = result.iteration_counts()
    early_average = np.mean(iteration_counts[:5])
    late_average = np.mean(iteration_counts[-5:])
    assert late_average > early_average


def test_documented_limitation_no_arc_length_method() -> None:
    """A load large enough to genuinely pass the limit point is not expected to
    converge under plain load control -- documenting, not attempting to fix,
    this known limitation (spec section 28: "if full snap-through cannot be
    robustly solved, document the limitation instead of creating an unstable
    implementation")."""
    from femtoolkit.exceptions import NonlinearConvergenceError

    mesh, truss_1, truss_2 = _build_shallow_truss_mesh()
    material = SaintVenantKirchhoff1D(youngs_modulus=YOUNGS_MODULUS)
    # A load well past this truss's limit point.
    settings = NonlinearSolverSettings(load_steps=10, tolerance=1e-9, max_iterations=30)
    analysis = NonlinearAnalysis(
        mesh, {truss_1.id: material, truss_2.id: material}, settings, geometric_nonlinearity=True
    )
    analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(3, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(3, Y, 0.0))
    analysis.add_load(NodalLoad(2, Y, -1.0e5))  # deliberately excessive

    with pytest.raises(NonlinearConvergenceError):
        analysis.solve()
