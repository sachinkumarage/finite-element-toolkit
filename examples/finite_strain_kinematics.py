"""Example: deformation gradient and Green-Lagrange strain, Version 16.

Demonstrates the core finite-strain kinematics
(femtoolkit.continuum.deformation) in isolation -- no elements or solver
involved -- for three prescribed cases: a uniaxial stretch, a rigid
rotation (the critical objectivity check: E must come out ~0), and a
combined stretch-plus-rotation, showing that Green-Lagrange strain
correctly reports only the physically meaningful stretch part.
"""

import numpy as np

from femtoolkit.continuum.deformation import (
    deformation_gradient,
    displacement_gradient,
    green_lagrange_strain_tensor,
    right_cauchy_green,
    validate_deformation_gradient,
)

# TET4 reference-configuration shape gradients for the unit right tetrahedron.
_REFERENCE_GRADIENTS = np.array(
    [[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
_REFERENCE_COORDS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def _report(label: str, current_coords: np.ndarray) -> None:
    displacements = (current_coords - _REFERENCE_COORDS).flatten()
    h = displacement_gradient(displacements, _REFERENCE_GRADIENTS)
    f = deformation_gradient(h)
    determinant = validate_deformation_gradient(f)
    c = right_cauchy_green(f)
    e = green_lagrange_strain_tensor(f)

    print(f"\n{label}")
    print("-" * len(label))
    print(f"F =\n{f}")
    print(f"det(F) = {determinant:.6f}  (volume ratio, current/reference)")
    print(f"C = F^T F =\n{c}")
    print(f"E = 1/2(C - I) =\n{e}")
    print(f"max|E| = {np.max(np.abs(e)):.3e}")


def main() -> None:
    """Report F, C, and E for a uniaxial stretch, a rigid rotation, and both combined."""
    print("Finite Element Toolkit")
    print("Version 16 -- Deformation Gradient and Green-Lagrange Strain")
    print("=" * 60)

    # Case 1: uniaxial stretch by 30% along X.
    stretch = _REFERENCE_COORDS.copy()
    stretch[:, 0] *= 1.3
    _report("Case 1: 30% uniaxial stretch along X", stretch)

    # Case 2: rigid rotation by 60 degrees about Z -- E must be ~0.
    theta = np.pi / 3
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]
    )
    rotated = _REFERENCE_COORDS @ rotation.T
    _report("Case 2: rigid 60-degree rotation about Z (objectivity check)", rotated)

    # Case 3: stretch then rotate -- E must match Case 1's stretch-only strain exactly
    # (rotation contributes nothing to E, by construction).
    stretched_and_rotated = stretch @ rotation.T
    _report("Case 3: 30% stretch followed by a rigid 60-degree rotation", stretched_and_rotated)


if __name__ == "__main__":
    main()
