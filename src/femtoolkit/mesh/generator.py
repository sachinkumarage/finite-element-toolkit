"""Structured 2D mesh generation.

Building a mesh by hand -- one :class:`~femtoolkit.mesh.node.Node` and one
:class:`~femtoolkit.mesh.cst_element.CSTElement2D` or
:class:`~femtoolkit.mesh.quad_element.QuadElement2D` call at a time --
does not scale past a handful of elements. This module automates the
common case: a rectangular domain subdivided into a regular grid of
cells, each becoming one Q4 element (:func:`create_quad_mesh`) or two CST
elements (:func:`create_triangular_mesh`).

.. code-block:: text

    Geometry (width, height, nx, ny)
        -> Mesh Generator
        -> Nodes + Elements
        -> Mesh (existing container, unchanged)
        -> Existing FEM solver

This is deliberately a **structured** mesh generator: it produces a
regular grid, not an arbitrary CAD-driven or unstructured mesh (that is
future-version scope). Its whole job is turning ``(width, height, nx,
ny)`` into correctly connected, correctly oriented, correctly numbered
:class:`~femtoolkit.mesh.mesh.Mesh` instances -- nothing more.

**Node numbering.** Row-major, bottom-to-top, left-to-right, 1-indexed:

.. code-block:: text

    node_id(row, col) = row * (nx + 1) + col + 1

    Node4 -------- Node5 -------- Node6      row 1 (top)
      |              |              |
      |              |              |
    Node1 -------- Node2 -------- Node3      row 0 (bottom)

    (nx=2, ny=1: node coordinates are (col * width/nx, row * height/ny))

This gives ``(nx + 1) * (ny + 1)`` nodes total, with node 1 always at the
domain's bottom-left corner.

**Element numbering.** Left-to-right within a row, then bottom-to-top
across rows, 1-indexed. For :func:`create_quad_mesh`, one element per
cell: ``element_id(row, col) = row * nx + col + 1``. For
:func:`create_triangular_mesh`, two elements per cell (see below), so
``element_id = 2 * (row * nx + col) + 1`` for the first triangle and
``+ 2`` for the second.

**Element node order.** Every generated element lists its nodes
counter-clockwise starting from the cell's bottom-left corner, matching
the orientation convention each element type already expects:
:class:`~femtoolkit.mesh.quad_element.QuadElement2D` requires
counter-clockwise order outright (see its module docstring), and
:class:`~femtoolkit.mesh.cst_element.CSTElement2D` accepts either winding
but this generator always emits counter-clockwise for consistency. The
generator therefore never produces a degenerate or inverted element --
enforced by calling :func:`~femtoolkit.mesh.validation.validate_mesh` on
its own output before returning (see the module's engineering
documentation in the project README for why this matters).
"""

from __future__ import annotations

import math
from typing import Literal

from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.mesh.node import Node
from femtoolkit.mesh.quad_element import QuadElement2D
from femtoolkit.mesh.validation import validate_mesh

DiagonalDirection = Literal["forward", "backward"]


def _validate_generator_inputs(width: float, height: float, nx: int, ny: int) -> None:
    if not math.isfinite(width) or width <= 0:
        raise ValidationError(f"width must be positive, got {width}.")
    if not math.isfinite(height) or height <= 0:
        raise ValidationError(f"height must be positive, got {height}.")
    if not isinstance(nx, int) or isinstance(nx, bool) or nx < 1:
        raise ValidationError(f"nx must be a positive integer, got {nx!r}.")
    if not isinstance(ny, int) or isinstance(ny, bool) or ny < 1:
        raise ValidationError(f"ny must be a positive integer, got {ny!r}.")


def _generate_grid_nodes(width: float, height: float, nx: int, ny: int) -> list[list[Node]]:
    """Build the (ny+1) x (nx+1) grid of nodes, indexed as ``grid[row][col]``."""
    dx = width / nx
    dy = height / ny
    grid: list[list[Node]] = []
    node_id = 1
    for row in range(ny + 1):
        grid_row = []
        for col in range(nx + 1):
            grid_row.append(Node(id=node_id, x=col * dx, y=row * dy, z=0.0))
            node_id += 1
        grid.append(grid_row)
    return grid


def create_quad_mesh(
    width: float,
    height: float,
    nx: int,
    ny: int,
    material: LinearElastic2D,
    thickness: float,
) -> Mesh:
    """Generate a structured rectangular mesh of Q4 elements.

    Args:
        width: Domain width (X extent), in meters. Must be positive.
        height: Domain height (Y extent), in meters. Must be positive.
        nx: Number of cell subdivisions along X. Must be a positive integer.
        ny: Number of cell subdivisions along Y. Must be a positive integer.
        material: The 2D constitutive model assigned to every generated
            element. The generator does not choose or default a material
            itself -- see the module docstring's note on keeping
            geometry and material concerns separate.
        thickness: Element thickness, in meters, assigned to every
            generated element. Must be positive.

    Returns:
        A :class:`~femtoolkit.mesh.mesh.Mesh` with ``(nx+1)*(ny+1)``
        nodes and ``nx*ny`` :class:`~femtoolkit.mesh.quad_element.QuadElement2D`
        elements.

    Raises:
        ValidationError: If ``width``, ``height``, ``nx``, or ``ny`` is
            not a positive (and, for ``nx``/``ny``, integer) value.

    Example:
        >>> mesh = create_quad_mesh(
        ...     width=2.0, height=1.0, nx=4, ny=2,
        ...     material=LinearElastic2D(200e9, 0.3, "plane_stress"),
        ...     thickness=0.01,
        ... )
        >>> len(mesh.nodes), len(mesh.elements)
        (15, 8)
    """
    _validate_generator_inputs(width, height, nx, ny)

    grid = _generate_grid_nodes(width, height, nx, ny)
    mesh = Mesh()
    for row in grid:
        for node in row:
            mesh.add_node(node)

    element_id = 1
    for row in range(ny):
        for col in range(nx):
            bottom_left = grid[row][col]
            bottom_right = grid[row][col + 1]
            top_right = grid[row + 1][col + 1]
            top_left = grid[row + 1][col]
            mesh.add_element(
                QuadElement2D(
                    id=element_id,
                    nodes=(bottom_left, bottom_right, top_right, top_left),
                    material=material,
                    thickness=thickness,
                )
            )
            element_id += 1

    validate_mesh(mesh)
    return mesh


def create_triangular_mesh(
    width: float,
    height: float,
    nx: int,
    ny: int,
    material: LinearElastic2D,
    thickness: float,
    diagonal: DiagonalDirection = "forward",
) -> Mesh:
    """Generate a structured mesh of CST elements by splitting each cell in two.

    Each rectangular cell becomes two triangles, split along one
    diagonal:

    .. code-block:: text

        diagonal="forward"  (bottom-left to top-right):

            top_left ------ top_right
               | \\             |
               |   \\           |
               |     \\         |
            bottom_left ---- bottom_right

            Triangle 1: (bottom_left, bottom_right, top_right)
            Triangle 2: (bottom_left, top_right, top_left)

        diagonal="backward"  (bottom-right to top-left):

            top_left ------ top_right
               |             /  |
               |           /    |
               |         /      |
            bottom_left ---- bottom_right

            Triangle 1: (bottom_left, bottom_right, top_left)
            Triangle 2: (bottom_right, top_right, top_left)

    Both triangles are always listed counter-clockwise. The diagonal
    direction is fixed per call (uniform across the whole mesh) --
    per-cell adaptive diagonal selection is out of scope for this
    version.

    Args:
        width: Domain width (X extent), in meters. Must be positive.
        height: Domain height (Y extent), in meters. Must be positive.
        nx: Number of cell subdivisions along X. Must be a positive integer.
        ny: Number of cell subdivisions along Y. Must be a positive integer.
        material: The 2D constitutive model assigned to every generated
            element.
        thickness: Element thickness, in meters, assigned to every
            generated element. Must be positive.
        diagonal: Which diagonal splits each cell, ``"forward"``
            (bottom-left to top-right, the default) or ``"backward"``
            (bottom-right to top-left).

    Returns:
        A :class:`~femtoolkit.mesh.mesh.Mesh` with ``(nx+1)*(ny+1)``
        nodes and ``2*nx*ny`` :class:`~femtoolkit.mesh.cst_element.CSTElement2D`
        elements.

    Raises:
        ValidationError: If ``width``, ``height``, ``nx``, or ``ny`` is
            not a positive (and, for ``nx``/``ny``, integer) value, or if
            ``diagonal`` is not ``"forward"`` or ``"backward"``.

    Example:
        >>> mesh = create_triangular_mesh(
        ...     width=2.0, height=1.0, nx=4, ny=2,
        ...     material=LinearElastic2D(200e9, 0.3, "plane_stress"),
        ...     thickness=0.01,
        ... )
        >>> len(mesh.nodes), len(mesh.elements)
        (15, 16)
    """
    _validate_generator_inputs(width, height, nx, ny)
    if diagonal not in ("forward", "backward"):
        raise ValidationError(f'diagonal must be "forward" or "backward", got {diagonal!r}.')

    grid = _generate_grid_nodes(width, height, nx, ny)
    mesh = Mesh()
    for row in grid:
        for node in row:
            mesh.add_node(node)

    element_id = 1
    for row in range(ny):
        for col in range(nx):
            bottom_left = grid[row][col]
            bottom_right = grid[row][col + 1]
            top_right = grid[row + 1][col + 1]
            top_left = grid[row + 1][col]

            if diagonal == "forward":
                triangle_1_nodes = (bottom_left, bottom_right, top_right)
                triangle_2_nodes = (bottom_left, top_right, top_left)
            else:
                triangle_1_nodes = (bottom_left, bottom_right, top_left)
                triangle_2_nodes = (bottom_right, top_right, top_left)

            mesh.add_element(
                CSTElement2D(
                    id=element_id, nodes=triangle_1_nodes, material=material, thickness=thickness
                )
            )
            mesh.add_element(
                CSTElement2D(
                    id=element_id + 1,
                    nodes=triangle_2_nodes,
                    material=material,
                    thickness=thickness,
                )
            )
            element_id += 2

    validate_mesh(mesh)
    return mesh
