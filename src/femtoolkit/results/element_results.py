"""Structured per-element result types.

:class:`~femtoolkit.mesh.frame_element.FrameElement2D` carries axial
force, shear force, and bending moment, and (unlike a bar or truss
element) these generally differ at its two ends. This module defines the
small, immutable data structures used to report that pair of results:
:class:`FrameEndForces` for one end, and :class:`FrameElementForces` for
both ends of one element.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameEndForces:
    """Internal forces at one end of a frame element.

    Attributes:
        axial_force: Axial (normal) force, in newtons. Positive is
            tension.
        shear_force: Shear force, in newtons, in the local transverse
            direction. See the sign convention documented in
            :mod:`femtoolkit.mesh.frame_element`.
        bending_moment: Bending moment, in newton-meters, about the local
            out-of-plane axis. See the sign convention documented in
            :mod:`femtoolkit.mesh.frame_element`.
    """

    axial_force: float
    shear_force: float
    bending_moment: float


@dataclass(frozen=True)
class FrameElementForces:
    """Internal forces at both ends of a frame element.

    Attributes:
        node_1: Forces at the element's first node.
        node_2: Forces at the element's second node.
    """

    node_1: FrameEndForces
    node_2: FrameEndForces
