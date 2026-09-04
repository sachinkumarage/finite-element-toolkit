"""Validation: nearly incompressible hyperelastic behavior and its known limits (spec section 21).

A high ``bulk_modulus`` relative to the shear-like parameters (``c10``,
``c01``) models a nearly incompressible rubber -- physically, volume
should barely change under load (``J`` approx ``1``) no matter how large
the applied deviatoric (shape-changing) load. This file validates that
behavior for a converged Newton-Raphson HEX8 solve (the physically
reachable, equilibrium regime), and separately documents -- as a passing,
explicit regression test rather than a comment -- the tangent-stability
finding from this project's own development: pushed to an *extreme*,
non-equilibrium kinematic probe (not a state any converged solve would
reach), the coupling between a very stiff volumetric term and shear can
locally produce a non-positive-definite material tangent. This is a
genuine, known limitation of displacement-based (non-mixed) finite
elements approaching incompressibility -- volumetric locking -- and is
*not* fixed here; a proper fix requires a mixed u-p formulation, out of
scope for this version (see the Version 17 section of the README).
"""

import numpy as np
import pytest

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.continuum.tensor import voigt_strain_to_tensor
from femtoolkit.materials import LinearElastic3D, MooneyRivlin3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node

X = TranslationDOF.X
Y = TranslationDOF.Y
Z = TranslationDOF.Z

_COORDS = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.0),
]
_FIXED_FACE = (1, 4, 5, 8)
_LOADED_FACE = (2, 3, 6, 7)


def test_high_bulk_modulus_keeps_jacobian_near_one_under_equilibrium_load() -> None:
    """A converged HEX8 solve with a stiff bulk modulus stays near-incompressible (J approx 1)."""
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    # bulk_modulus >> c10 + c01: a nearly incompressible rubber (Poisson ratio -> 0.5).
    material = MooneyRivlin3D(c10=0.4e6, c01=0.1e6, bulk_modulus=2000e6)
    settings = NonlinearSolverSettings(load_steps=20, tolerance=1e-8, max_iterations=40)
    analysis = NonlinearAnalysis(mesh, {hexa.id: material}, settings, geometric_nonlinearity=True)
    for node_id in _FIXED_FACE:
        for dof in (X, Y, Z):
            analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))
    load_per_node = 3.0e5
    for node_id in _LOADED_FACE:
        analysis.add_load(NodalLoad(node_id, X, load_per_node))

    result = analysis.solve()
    assert result.converged

    for gauss_point in range(8):
        state = result.element_state(1, gauss_point=gauss_point)
        strain_tensor = voigt_strain_to_tensor(state.strain)
        c = 2.0 * strain_tensor + np.eye(3)
        jacobian = np.sqrt(np.linalg.det(c))
        assert jacobian == pytest.approx(1.0, abs=0.01)


def test_higher_bulk_modulus_gives_smaller_volume_change_than_lower() -> None:
    """Increasing bulk_modulus (at fixed load) should push J closer to 1 -- a basic
    sanity check that the volumetric penalty term is actually doing its job."""
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))

    def _max_volume_deviation(bulk_modulus: float) -> float:
        hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)
        mesh = Mesh()
        for node in nodes:
            mesh.add_node(node)
        mesh.add_element(hexa)

        material = MooneyRivlin3D(c10=0.4e6, c01=0.1e6, bulk_modulus=bulk_modulus)
        settings = NonlinearSolverSettings(load_steps=20, tolerance=1e-8, max_iterations=40)
        analysis = NonlinearAnalysis(
            mesh, {hexa.id: material}, settings, geometric_nonlinearity=True
        )
        for node_id in _FIXED_FACE:
            for dof in (X, Y, Z):
                analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))
        load_per_node = 3.0e5
        for node_id in _LOADED_FACE:
            analysis.add_load(NodalLoad(node_id, X, load_per_node))
        result = analysis.solve()
        assert result.converged

        deviations = []
        for gauss_point in range(8):
            state = result.element_state(1, gauss_point=gauss_point)
            strain_tensor = voigt_strain_to_tensor(state.strain)
            c = 2.0 * strain_tensor + np.eye(3)
            jacobian = np.sqrt(np.linalg.det(c))
            deviations.append(abs(jacobian - 1.0))
        return max(deviations)

    soft_deviation = _max_volume_deviation(bulk_modulus=20e6)
    stiff_deviation = _max_volume_deviation(bulk_modulus=2000e6)
    assert stiff_deviation < soft_deviation


def test_extreme_volumetric_shear_coupling_can_locally_break_tangent_pd() -> None:
    """Documents (as a passing regression test) a known limitation found during development.

    At an extreme, non-equilibrium kinematic probe -- combining shear with
    a modest volume change, under a bulk modulus hundreds of times stiffer
    than the shear-like parameters -- the material tangent can be locally
    non-positive-definite. This was confirmed (via linearization-accuracy
    checks) to be a genuine property of the implemented energy function at
    that extreme state, not a numerical-differentiation artifact -- see
    femtoolkit.materials.hyperelastic's module docstring. A converged
    Newton-Raphson solve never reaches this state (see the two tests
    above); this test exists to document the limitation explicitly rather
    than leave it undiscoverable.
    """
    from femtoolkit.materials import MooneyRivlin3D as _MooneyRivlin3D

    material = _MooneyRivlin3D(c10=0.4e6, c01=0.1e6, bulk_modulus=200e6)
    f = np.diag([1.3, 0.9, 0.95])  # combined shear-adjacent, non-isochoric probe
    tangent = material.material_tangent(f)
    eigenvalues = np.linalg.eigvalsh(tangent)
    assert np.any(eigenvalues < 0), (
        "Expected this documented extreme-state tangent instability to still be "
        "present; if this now passes, either the material model or the finite "
        "difference scheme changed and this test (and the README's Version 17 "
        "Limitations section) should be revisited."
    )
