"""Tests for JSON mesh export/import (femtoolkit.mesh.serialization)."""

import json

import pytest
from numpy.testing import assert_allclose

from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import LinearElastic2D, Material
from femtoolkit.mesh import (
    CSTElement2D,
    Mesh,
    Node,
    create_quad_mesh,
    create_triangular_mesh,
    export_mesh,
    import_mesh,
    mesh_from_dict,
    mesh_to_dict,
)


@pytest.fixture
def material() -> LinearElastic2D:
    return LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")


def test_mesh_to_dict_schema(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=2, ny=1, material=material, thickness=0.01)

    data = mesh_to_dict(mesh)

    assert set(data.keys()) == {"metadata", "nodes", "elements"}
    assert data["metadata"]["node_count"] == 6
    assert data["metadata"]["element_count"] == 2
    assert len(data["nodes"]) == 6
    assert len(data["elements"]) == 2
    assert data["elements"][0]["type"] == "QuadElement2D"
    assert data["elements"][0]["node_ids"] == [1, 2, 5, 4]


def test_mesh_to_dict_is_json_serializable(material: LinearElastic2D) -> None:
    mesh = create_triangular_mesh(
        width=1.0, height=1.0, nx=2, ny=2, material=material, thickness=0.01
    )

    data = mesh_to_dict(mesh)
    serialized = json.dumps(data)  # must not raise

    assert isinstance(serialized, str)


def test_round_trip_quad_mesh_nodes_match(material: LinearElastic2D) -> None:
    original = create_quad_mesh(
        width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01
    )

    data = mesh_to_dict(original)
    restored = mesh_from_dict(data)

    assert len(restored.nodes) == len(original.nodes)
    for original_node, restored_node in zip(original.nodes, restored.nodes, strict=True):
        assert original_node.id == restored_node.id
        assert_allclose((original_node.x, original_node.y), (restored_node.x, restored_node.y))


def test_round_trip_quad_mesh_elements_match(material: LinearElastic2D) -> None:
    original = create_quad_mesh(
        width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01
    )

    restored = mesh_from_dict(mesh_to_dict(original))

    assert len(restored.elements) == len(original.elements)
    pairs = zip(original.elements, restored.elements, strict=True)
    for original_element, restored_element in pairs:
        assert original_element.id == restored_element.id
        assert [n.id for n in original_element.nodes] == [n.id for n in restored_element.nodes]
        assert_allclose(restored_element.area, original_element.area)
        assert restored_element.thickness == original_element.thickness
        assert restored_element.material.youngs_modulus == original_element.material.youngs_modulus
        assert restored_element.material.poisson_ratio == original_element.material.poisson_ratio
        assert restored_element.material.formulation == original_element.material.formulation


def test_round_trip_triangular_mesh(material: LinearElastic2D) -> None:
    original = create_triangular_mesh(
        width=2.0, height=1.0, nx=3, ny=2, material=material, thickness=0.02
    )

    restored = mesh_from_dict(mesh_to_dict(original))

    assert len(restored.nodes) == len(original.nodes)
    assert len(restored.elements) == len(original.elements)
    assert_allclose(
        sum(e.area for e in restored.elements), sum(e.area for e in original.elements)
    )


def test_round_trip_metadata_matches(material: LinearElastic2D) -> None:
    original = create_quad_mesh(
        width=3.0, height=2.0, nx=3, ny=2, material=material, thickness=0.01
    )

    data = mesh_to_dict(original)
    assert data["metadata"]["node_count"] == len(original.nodes)
    assert data["metadata"]["element_count"] == len(original.elements)

    restored = mesh_from_dict(data)
    restored_data = mesh_to_dict(restored)
    assert restored_data["metadata"] == data["metadata"]


def test_export_and_import_file_round_trip(material: LinearElastic2D, tmp_path) -> None:
    original = create_quad_mesh(
        width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01
    )
    path = tmp_path / "mesh.json"

    export_mesh(original, path)
    assert path.exists()

    restored = import_mesh(path)
    assert len(restored.nodes) == len(original.nodes)
    assert len(restored.elements) == len(original.elements)
    assert_allclose(
        sum(e.area for e in restored.elements), sum(e.area for e in original.elements)
    )


def test_export_produces_valid_json_file(material: LinearElastic2D, tmp_path) -> None:
    mesh = create_quad_mesh(width=1.0, height=1.0, nx=2, ny=2, material=material, thickness=0.01)
    path = tmp_path / "mesh.json"

    export_mesh(mesh, path)

    with path.open() as file:
        data = json.load(file)
    assert data["metadata"]["node_count"] == 9


def test_mesh_to_dict_rejects_unsupported_element_type() -> None:
    from femtoolkit.mesh import Element

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    bar_material = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    mesh.add_element(Element(id=1, nodes=[node_1, node_2], material=bar_material))

    with pytest.raises(ValidationError):
        mesh_to_dict(mesh)


def test_mesh_from_dict_rejects_unsupported_element_type() -> None:
    data = {
        "metadata": {"format_version": "1.0", "node_count": 2, "element_count": 1},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
            {"id": 2, "x": 1.0, "y": 0.0, "z": 0.0},
        ],
        "elements": [
            {
                "id": 1,
                "type": "BarElement",
                "node_ids": [1, 2],
                "thickness": 0.01,
                "material": {
                    "youngs_modulus": 200e9,
                    "poisson_ratio": 0.3,
                    "formulation": "plane_stress",
                },
            }
        ],
    }

    with pytest.raises(ValidationError):
        mesh_from_dict(data)


def test_mesh_with_mixed_materials_round_trips_correctly() -> None:
    steel = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    aluminum = LinearElastic2D(youngs_modulus=70e9, poisson_ratio=0.33, formulation="plane_stress")

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=0.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=1.0, y=1.0, z=0.0)
    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4):
        mesh.add_node(node)
    mesh.add_element(
        CSTElement2D(id=1, nodes=(node_1, node_2, node_3), material=steel, thickness=0.01)
    )
    mesh.add_element(
        CSTElement2D(id=2, nodes=(node_2, node_4, node_3), material=aluminum, thickness=0.02)
    )

    restored = mesh_from_dict(mesh_to_dict(mesh))

    assert restored.get_element(1).material.youngs_modulus == steel.youngs_modulus
    assert restored.get_element(2).material.youngs_modulus == aluminum.youngs_modulus
    assert restored.get_element(1).thickness == 0.01
    assert restored.get_element(2).thickness == 0.02
