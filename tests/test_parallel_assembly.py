"""Tests for femtoolkit.analysis.parallel_assembly (Version 27)."""

from __future__ import annotations

from numpy.testing import assert_allclose

from femtoolkit.analysis.assembly import assemble_global_stiffness
from femtoolkit.analysis.dof import DOFMap
from femtoolkit.analysis.parallel_assembly import (
    compute_conductivity_contributions,
    compute_stiffness_contributions,
)
from femtoolkit.execution import ExecutionConfig, create_executor
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh.generator import create_quad_mesh
from femtoolkit.thermal.thermal_material import ThermalMaterial


def _mechanical_mesh():
    material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    return create_quad_mesh(width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01)


def _thermal_mesh_and_materials():
    dummy = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    mesh = create_quad_mesh(width=1.0, height=1.0, nx=3, ny=3, material=dummy, thickness=0.01)
    thermal_material = ThermalMaterial(
        thermal_conductivity=45.0, density=7850.0, specific_heat=460.0
    )
    materials = {element.id: thermal_material for element in mesh.elements}
    return mesh, materials


# --- compute_stiffness_contributions ----------------------------------------


def test_compute_stiffness_contributions_serial_matches_direct_loop() -> None:
    mesh = _mechanical_mesh()
    executor = create_executor(ExecutionConfig(mode="serial"))

    contributions = compute_stiffness_contributions(mesh.elements, executor)

    assert len(contributions) == len(mesh.elements)
    for contribution, element in zip(contributions, mesh.elements, strict=True):
        assert contribution.dof_keys == element.dof_keys()
        assert_allclose(contribution.stiffness, element.stiffness_matrix)


def test_compute_stiffness_contributions_parallel_matches_serial() -> None:
    mesh = _mechanical_mesh()
    dof_map = DOFMap(node_ids=[n.id for n in mesh.nodes], dofs_per_node=2)

    serial = compute_stiffness_contributions(
        mesh.elements, create_executor(ExecutionConfig(mode="serial"))
    )
    parallel = compute_stiffness_contributions(
        mesh.elements,
        create_executor(ExecutionConfig(mode="parallel", workers=2, backend="process")),
    )

    k_serial = assemble_global_stiffness(dof_map, serial)
    k_parallel = assemble_global_stiffness(dof_map, parallel)
    assert_allclose(k_serial, k_parallel)


def test_compute_stiffness_contributions_preserves_element_order() -> None:
    mesh = _mechanical_mesh()
    executor = create_executor(ExecutionConfig(mode="parallel", workers=2, backend="process"))

    contributions = compute_stiffness_contributions(mesh.elements, executor)

    for contribution, element in zip(contributions, mesh.elements, strict=True):
        assert contribution.dof_keys == element.dof_keys()


def test_compute_stiffness_contributions_empty_mesh_elements() -> None:
    executor = create_executor(ExecutionConfig(mode="serial"))
    assert compute_stiffness_contributions([], executor) == []


# --- compute_conductivity_contributions -------------------------------------


def test_compute_conductivity_contributions_serial_matches_direct_call() -> None:
    from femtoolkit.thermal.thermal_elements import conductivity_contribution

    mesh, materials = _thermal_mesh_and_materials()
    executor = create_executor(ExecutionConfig(mode="serial"))

    contributions = compute_conductivity_contributions(mesh.elements, materials, 293.15, executor)

    for contribution, element in zip(contributions, mesh.elements, strict=True):
        expected = conductivity_contribution(element, materials[element.id], 293.15)
        assert contribution.dof_keys == expected.dof_keys
        assert_allclose(contribution.stiffness, expected.stiffness)


def test_compute_conductivity_contributions_parallel_matches_serial() -> None:
    mesh, materials = _thermal_mesh_and_materials()
    dof_map = DOFMap(node_ids=[n.id for n in mesh.nodes], dofs_per_node=1)

    serial = compute_conductivity_contributions(
        mesh.elements, materials, 293.15, create_executor(ExecutionConfig(mode="serial"))
    )
    parallel = compute_conductivity_contributions(
        mesh.elements,
        materials,
        293.15,
        create_executor(ExecutionConfig(mode="parallel", workers=2, backend="process")),
    )

    k_serial = assemble_global_stiffness(dof_map, serial)
    k_parallel = assemble_global_stiffness(dof_map, parallel)
    assert_allclose(k_serial, k_parallel)


def test_compute_conductivity_contributions_empty_elements() -> None:
    executor = create_executor(ExecutionConfig(mode="serial"))
    assert compute_conductivity_contributions([], {}, 293.15, executor) == []
