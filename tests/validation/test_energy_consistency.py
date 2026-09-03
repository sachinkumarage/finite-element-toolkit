"""Validation: external work vs. internal strain energy for a conservative system (spec section 26).

For a hyperelastic (path-independent, conservative) material under
quasi-static load, the work done by the external load must equal the
strain energy stored internally: ``W_ext = U_internal``. For St. Venant-
Kirchhoff elasticity (``S = C : E``, linear), the internal strain energy
has the closed form ``U = 1/2 * V0 * E : S`` (a perfect differential of the
linear ``S(E)`` relationship, so this is exact regardless of the loading
path). External work is *not* simply ``1/2 * F * u`` here, since the
load-displacement relationship itself is nonlinear once geometric effects
matter -- it is instead the trapezoidal integral of external force against
displacement across the recorded load-step history, which
:class:`~femtoolkit.results.nonlinear_result.NonlinearAnalysisResult`
already provides in full.
"""

import numpy as np
import pytest

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.continuum.tensor import voigt_strain_to_tensor, voigt_stress_to_tensor
from femtoolkit.materials import LinearElastic3D, SaintVenantKirchhoff3D
from femtoolkit.mesh import Mesh, Node, Tet4Element3D

X = TranslationDOF.X
Y = TranslationDOF.Y
Z = TranslationDOF.Z


def test_tet4_external_work_matches_internal_strain_energy() -> None:
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=0.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=0.0, z=1.0)
    tet = Tet4Element3D(id=1, nodes=(node_1, node_2, node_3, node_4), material=placeholder)

    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4):
        mesh.add_node(node)
    mesh.add_element(tet)

    material = SaintVenantKirchhoff3D(youngs_modulus=200e9, poisson_ratio=0.3)
    settings = NonlinearSolverSettings(load_steps=25, tolerance=1e-10, max_iterations=40)
    analysis = NonlinearAnalysis(mesh, {tet.id: material}, settings, geometric_nonlinearity=True)
    for node_id in (1, 3, 4):
        for dof in (X, Y, Z):
            analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))
    applied_load = 5.0e7
    analysis.add_load(NodalLoad(2, X, applied_load))

    result = analysis.solve()
    assert result.converged

    # External work: trapezoidal integral of external force against displacement
    # across the recorded load-step history (node 2's X DOF is the only one with
    # nonzero external load; every other DOF's external force is always zero).
    displacement_history = result.displacement_history(2, X)
    load_factors = result.load_factors()
    force_history = load_factors * applied_load

    external_work = np.trapezoid(force_history, displacement_history)

    # Internal strain energy: U = 1/2 * V0 * E:S (exact for St. Venant-Kirchhoff).
    final_strain_voigt = result.element_strain(1)
    final_stress_voigt = result.element_stress(1)
    strain_tensor = voigt_strain_to_tensor(final_strain_voigt)
    stress_tensor = voigt_stress_to_tensor(final_stress_voigt)
    internal_energy = 0.5 * tet.volume * float(np.tensordot(strain_tensor, stress_tensor))

    assert external_work == pytest.approx(internal_energy, rel=5e-2)
