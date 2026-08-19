"""Validation: uniform thermal expansion of a whole rectangular plate.

A rectangular Q4-meshed plate, supported only by a pin at the bottom-left
corner and a roller (Y only) at the bottom-right corner -- the same
statically determinate support pattern validated in Version 9 -- is free
to expand under a uniform temperature change without developing any
stress. The analytical solution is simple rigid-body-free thermal
expansion about the pin:

.. code-block:: text

    ux(x, y) = alpha * dT * x
    uy(x, y) = alpha * dT * y

and every element's thermal-corrected stress (see
:mod:`femtoolkit.analysis.thermal_load`) must be zero.
"""

import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, StaticLinearAnalysis, TranslationDOF
from femtoolkit.analysis.thermal_load import (
    TemperatureLoad,
    thermal_corrected_stress,
    thermal_load_to_nodal_loads,
)
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import create_quad_mesh

YOUNGS_MODULUS = 200e9
POISSON_RATIO = 0.3
ALPHA = 12e-6
DELTA_T = 80.0
THICKNESS = 0.01
WIDTH = 2.0
HEIGHT = 1.0


@pytest.fixture
def result_and_mesh():
    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS,
        poisson_ratio=POISSON_RATIO,
        formulation="plane_stress",
        thermal_expansion_coefficient=ALPHA,
    )
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=4, ny=2, material=material, thickness=THICKNESS
    )

    thermal_load = TemperatureLoad(delta_temperature=DELTA_T)
    loads = thermal_load_to_nodal_loads(mesh, thermal_load)

    pin_node = min(mesh.nodes, key=lambda n: (n.x, n.y))
    roller_node = max((n for n in mesh.nodes if n.y == 0.0), key=lambda n: n.x)

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(pin_node.id, TranslationDOF.X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(pin_node.id, TranslationDOF.Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(roller_node.id, TranslationDOF.Y, 0.0))
    for load in loads:
        analysis.add_load(load)

    return analysis.solve(), mesh, thermal_load


def test_every_node_matches_rigid_body_free_expansion(result_and_mesh) -> None:
    result, mesh, _ = result_and_mesh

    for node in mesh.nodes:
        ux, uy = result.node_displacement(node.id)
        assert_allclose(ux, ALPHA * DELTA_T * node.x, atol=1e-12)
        assert_allclose(uy, ALPHA * DELTA_T * node.y, atol=1e-12)


def test_every_element_has_zero_thermal_corrected_stress(result_and_mesh) -> None:
    result, mesh, thermal_load = result_and_mesh

    for element in mesh.elements:
        corrected = thermal_corrected_stress(result, element, thermal_load)
        assert_allclose(corrected, [0.0, 0.0, 0.0], atol=1e-2)
