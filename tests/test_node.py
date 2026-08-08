"""Tests for the Node data model."""

import math

import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.mesh import Node


def test_valid_node_creation() -> None:
    node = Node(id=1, x=1.0, y=2.0, z=3.0)

    assert node.id == 1
    assert node.x == 1.0
    assert node.y == 2.0
    assert node.z == 3.0


def test_node_default_origin_coordinates() -> None:
    node = Node(id=1, x=0.0, y=0.0, z=0.0)

    assert (node.x, node.y, node.z) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("node_id", [0, -1, 1.5, "1"])
def test_invalid_id_raises(node_id) -> None:
    with pytest.raises(ValidationError):
        Node(id=node_id, x=0.0, y=0.0, z=0.0)


@pytest.mark.parametrize(
    "x, y, z",
    [
        (math.nan, 0.0, 0.0),
        (0.0, math.inf, 0.0),
        (0.0, 0.0, -math.inf),
    ],
)
def test_invalid_coordinates_raise(x: float, y: float, z: float) -> None:
    with pytest.raises(ValidationError):
        Node(id=1, x=x, y=y, z=z)
